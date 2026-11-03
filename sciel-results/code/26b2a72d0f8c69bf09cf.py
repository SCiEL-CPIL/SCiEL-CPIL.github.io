"""Improved packer for 300 equal circles in a unit circle."""
import time
import numpy as np

N = 300
DEADLINE = 150.0  # wall-clock limit for the evaluator


def _hex_lattice_init(n):
    """Generate a hexagonal lattice point cloud, return first n points inside unit disk."""
    # Estimate spacing so we get roughly n points in unit disk
    # Area of unit disk = pi, area per hex cell ~ sqrt(3)/2 * s^2 where s is spacing
    # n * sqrt(3)/2 * s^2 ~ pi  =>  s ~ sqrt(2*pi/(n*sqrt(3)/2)) = sqrt(4*pi/(n*sqrt(3)))
    s = np.sqrt(4.0 * np.pi / (n * np.sqrt(3.0)))
    # Use a slightly smaller spacing to ensure enough points
    s *= 0.95
    
    k = int(np.ceil(1.0 / s)) + 2
    pts = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 1.0:
                pts.append((x, y))
    
    pts = np.array(pts)
    if len(pts) < n:
        # Fallback: random
        theta = np.random.uniform(0, 2*np.pi, n)
        r = np.sqrt(np.random.uniform(0, 1, n))
        return np.column_stack([r*np.cos(theta), r*np.sin(theta)])
    
    # Sort by distance from origin, take closest n
    dist = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
    idx = np.argsort(dist, kind='stable')[:n]
    return pts[idx]


def _compute_radius(pts):
    """Compute the maximum common radius for the given centers."""
    if len(pts) == 0:
        return 0.0
    # Min distance to boundary
    dist_to_boundary = 1.0 - np.sqrt(np.sum(pts**2, axis=1))
    r_boundary = np.min(dist_to_boundary)
    
    # Min pairwise distance / 2
    # Use scipy's cKDTree for efficiency
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    # Query nearest neighbor for each point
    dists, _ = tree.query(pts, k=2)
    min_pair_dist = np.min(dists[:, 1])
    r_pair = min_pair_dist / 2.0
    
    return min(r_boundary, r_pair)


def _optimize(pts, time_budget=10.0):
    """Local optimization using gradient ascent on the radius objective."""
    start = time.time()
    pts = pts.astype(np.float64).copy()
    n = len(pts)
    
    # We want to maximize r = min(min_pairwise/2, min_boundary)
    # Equivalent to: for each constraint, push points apart / toward center
    # Use a simple iterative approach: repeatedly move each point to increase the min constraint
    
    best_r = _compute_radius(pts)
    
    # Gradient-based refinement
    # For each point, the gradient of the objective w.r.t. that point
    # depends on which constraint is active
    
    from scipy.spatial import cKDTree
    
    step_size = 0.001
    max_iters = 5000
    
    for it in range(max_iters):
        if time.time() - start > time_budget:
            break
        
        # Compute current radius
        dist_to_boundary = 1.0 - np.sqrt(np.sum(pts**2, axis=1))
        
        tree = cKDTree(pts)
        dists, nn_idx = tree.query(pts, k=2)
        nn_dist = dists[:, 1]
        nn_idx = nn_idx[:, 1]
        
        # The active constraints:
        # 1. Points near boundary: those with small dist_to_boundary
        # 2. Closest pairs: those with small nn_dist
        
        r_boundary = np.min(dist_to_boundary)
        r_pair = np.min(nn_dist) / 2.0
        r = min(r_boundary, r_pair)
        
        if r <= best_r:
            # Try smaller step
            step_size *= 0.9
            if step_size < 1e-8:
                break
        else:
            best_r = r
            step_size = min(step_size * 1.05, 0.01)
        
        # Compute forces
        forces = np.zeros_like(pts)
        
        # Boundary force: push points inward if they're near boundary
        # For points where dist_to_boundary is close to r_boundary
        boundary_active = dist_to_boundary < r_boundary + 0.01
        for i in np.where(boundary_active)[0]:
            norm = np.sqrt(pts[i, 0]**2 + pts[i, 1]**2)
            if norm > 1e-10:
                # Push toward center
                forces[i] -= (pts[i] / norm) * step_size
        
        # Pairwise forces: push apart closest pairs
        pair_active = nn_dist < 2.0 * r_pair + 0.01
        for i in np.where(pair_active)[0]:
            j = nn_idx[i]
            d = pts[i] - pts[j]
            dist = np.sqrt(np.sum(d**2))
            if dist > 1e-10:
                # Push i away from j
                forces[i] += (d / dist) * step_size
                forces[j] -= (d / dist) * step_size
        
        pts += forces
        
        # Ensure points stay inside unit circle
        norms = np.sqrt(np.sum(pts**2, axis=1))
        outside = norms > 1.0
        if np.any(outside):
            pts[outside] *= (1.0 / norms[outside])[:, np.newaxis]
    
    return pts, best_r


def _refine_with_scipy(pts, time_budget=20.0):
    """Use scipy minimize to refine the packing."""
    start = time.time()
    pts = pts.astype(np.float64).copy()
    n = len(pts)
    
    # Flatten to 1D
    x0 = pts.flatten()
    
    def objective(x):
        if time.time() - start > time_budget:
            return 0.0
        p = x.reshape(n, 2)
        # Negative radius (we minimize negative)
        dist_to_boundary = 1.0 - np.sqrt(np.sum(p**2, axis=1))
        r_boundary = np.min(dist_to_boundary)
        
        from scipy.spatial import cKDTree
        tree = cKDTree(p)
        dists, _ = tree.query(p, k=2)
        min_pair = np.min(dists[:, 1])
        r_pair = min_pair / 2.0
        
        return -min(r_boundary, r_pair)
    
    # Use Nelder-Mead or Powell for derivative-free optimization
    # But with 600 dimensions, this is too slow. Use a subset approach.
    
    # Instead, do coordinate-wise local search
    best_x = x0.copy()
    best_obj = objective(best_x)
    
    # Random restarts with small perturbations
    np.random.seed(42)
    for restart in range(50):
        if time.time() - start > time_budget:
            break
        
        # Perturb a few random points
        n_perturb = max(1, n // 10)
        idx = np.random.choice(n, n_perturb, replace=False)
        x_trial = best_x.copy()
        perturbation = np.random.normal(0, 0.005, (n_perturb, 2))
        x_trial[idx*2] += perturbation[:, 0]
        x_trial[idx*2 + 1] += perturbation[:, 1]
        
        # Ensure inside unit circle
        p_trial = x_trial.reshape(n, 2)
        norms = np.sqrt(np.sum(p_trial**2, axis=1))
        outside = norms > 1.0
        if np.any(outside):
            p_trial[outside] *= (0.99 / norms[outside])[:, np.newaxis]
            x_trial = p_trial.flatten()
        
        obj_trial = objective(x_trial)
        if obj_trial < best_obj:
            best_x = x_trial
            best_obj = obj_obj = obj_trial
    
    return best_x.reshape(n, 2), -best_obj


def construct():
    start_time = time.time()
    
    # Initial hexagonal lattice
    pts = _hex_lattice_init(N)
    
    # Quick local optimization
    pts, r1 = _optimize(pts, time_budget=30.0)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 100.0:
        return pts
    
    # Further refinement
    pts, r2 = _refine_with_scipy(pts, time_budget=40.0)
    
    elapsed = time.time() - start_time
    if elapsed > 140.0:
        return pts
    
    # Final polish: one more round of local optimization
    pts, r3 = _optimize(pts, time_budget=5.0)
    
    # Ensure validity: all points inside unit circle
    norms = np.sqrt(np.sum(pts**2, axis=1))
    pts[norms > 1.0] *= (1.0 / norms[norms > 1.0])[:, np.newaxis]
    
    # Final check
    final_r = _compute_radius(pts)
    
    return pts
