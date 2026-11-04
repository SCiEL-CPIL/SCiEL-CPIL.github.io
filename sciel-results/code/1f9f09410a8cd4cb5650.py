"""Pack 300 equal circles in the unit disk, maximizing the common radius."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 300
DEADLINE = 142.0  # safety margin before 150s kill
START_TIME = time.time()


def _elapsed():
    return time.time() - START_TIME


def _remaining():
    return max(0.0, DEADLINE - _elapsed())


def _compute_radius(pts):
    """Compute the packing radius for a set of centers."""
    n = len(pts)
    if n < 2:
        return float(np.min(1.0 - np.linalg.norm(pts, axis=1)))
    boundary = 1.0 - np.linalg.norm(pts, axis=1)
    min_boundary = np.min(boundary)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(min_boundary, min_pair)


def _soft_min(x, k):
    """Soft minimum via log-sum-exp."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return 0.0
    x_max = np.max(x)
    shifted = x - x_max
    log_sum = np.log(np.sum(np.exp(-k * shifted)))
    return x_max - log_sum / k


def _hex_init(seed):
    """Generate initial configuration using hexagonal lattice."""
    rng = np.random.default_rng(seed)
    
    # Target spacing for 300 circles
    # Area per circle ~ pi/300, so spacing ~ sqrt(pi/300) ~ 0.102
    # But we need to fit in unit disk, so let's try different spacings
    best_pts = None
    best_count = 0
    
    for spacing in np.arange(0.055, 0.11, 0.002):
        # Generate hex lattice points
        dx = spacing
        dy = spacing * np.sqrt(3) / 2
        points = []
        # Range to cover unit disk
        extent = int(1.0 / dx) + 2
        for j in range(-extent, extent + 1):
            for i in range(-extent, extent + 1):
                x = i * dx + (j % 2) * dx / 2
                y = j * dy
                if x * x + y * y <= (1.0 - spacing / 2) ** 2:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) >= N:
            # Select N points closest to center
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists, kind="stable")[:N]
            pts = points[idx].copy()
            # Add small random perturbation
            pts += rng.normal(0, spacing * 0.05, size=pts.shape)
            # Ensure inside disk
            dists = np.linalg.norm(pts, axis=1)
            mask = dists > 1.0
            if np.any(mask):
                pts[mask] /= dists[mask, np.newaxis] * (1.0 + 1e-8)
            
            r = _compute_radius(pts)
            if best_pts is None or r > best_count:
                best_pts = pts
                best_count = r
    
    if best_pts is None:
        # Fallback: random points
        angles = rng.uniform(0, 2 * np.pi, N)
        radii = rng.uniform(0, 0.95, N)
        best_pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    return best_pts


def _active_edges(pts, k=10):
    """Find active edges (pairs that are close to each other)."""
    tree = cKDTree(pts)
    # Query for k nearest neighbors
    dists, indices = tree.query(pts, k=k + 1)
    # dists[:, 0] is distance to self (0)
    # Build set of active pairs
    active = set()
    for i in range(N):
        for j_idx in range(1, k + 1):
            j = indices[i, j_idx]
            d = dists[i, j_idx]
            if d < 0.15:  # threshold for "active"
                active.add((min(i, j), max(i, j)))
    return list(active)


def _objective_and_grad(params, k_soft=50.0, active_edges=None):
    """
    Objective: minimize -soft_min(boundary, half_pair_dists)
    Gradient computed analytically for soft-min.
    """
    pts = params.reshape(-1, 2)
    dist_to_center = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - dist_to_center
    
    # Pairwise distances for active edges only (or all if not provided)
    if active_edges is None:
        active_edges = _active_edges(pts)
    
    # Collect all distances
    all_dists = list(boundary)
    all_grad_indices = []  # (dist_index, point_index, component) for gradient
    all_grad_weights = []  # weight for each point-component
    
    # Initialize gradient accumulator
    grad = np.zeros_like(pts)
    
    # We'll compute soft_min over all distances
    # For efficiency, let's just use all pairwise distances for now
    # Since N=300, 300*299/2 = 44850 pairs, manageable
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    half_dists = dists / 2.0
    
    # Combine boundary and half_dists
    all_dists_vec = np.concatenate([boundary, half_dists.ravel()])
    
    # Compute soft_min and weights
    x_max = np.max(all_dists_vec)
    shifted = all_dists_vec - x_max
    exp_terms = np.exp(-k_soft * shifted)
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp
    
    soft_min_val = x_max - np.log(sum_exp) / k_soft
    
    # Gradient of soft_min w.r.t. each distance
    # d(soft_min)/d(x_i) = weights[i]
    # d(boundary_i)/d(pts_i) = -pts_i / |pts_i|
    # d(half_dist_ij)/d(pts_i) = -0.5 * (pts_i - pts_j) / |pts_i - pts_j|
    # d(half_dist_ij)/d(pts_j) = +0.5 * (pts_i - pts_j) / |pts_i - pts_j|
    
    # Boundary gradient
    for i in range(N):
        d = dist_to_center[i]
        if d > 1e-10:
            grad[i] -= weights[i] * pts[i] / d
    
    # Pairwise gradient
    # For each pair (i, j), i < j
    # weight = weights[N + i*N + j] (since half_dists is N x N, flattened)
    # Actually half_dists.ravel() has index i*N + j for element (i, j)
    # But we only want i < j to avoid double counting
    # The dists matrix is symmetric, so half_dists[i,j] = half_dists[j,i]
    # In the flattened array, index for (i,j) is i*N + j
    # We need to be careful: the gradient contribution from pair (i,j) appears in both
    # weights[i*N+j] and weights[j*N+i]
    
    # Let's compute more efficiently
    # For each pair i < j:
    # w_ij = weights[i*N + j] + weights[j*N + i]  (both contribute)
    # Actually no: soft_min is over the full vector, so each element has its own weight.
    # The gradient w.r.t. pts_i gets contribution from:
    # - boundary_i: weight = weights[i]
    # - half_dist_ij for all j != i: weight = weights[i*N + j] (if we consider dists[i,j])
    #   But dists[i,j] = dists[j,i], and in the flattened half_dists, 
    #   element (i,j) is at index i*N+j, element (j,i) is at index j*N+i.
    #   Both are in the vector.
    
    # So for point i, the gradient from pairs is:
    # sum_{j != i} weights[i*N + j] * d(half_dist_ij)/d(pts_i)
    # = sum_{j != i} weights[i*N + j] * (-0.5) * (pts_i - pts_j) / dists[i,j]
    
    # This is O(N^2) which is fine for N=300.
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            w = weights[i * N + j]
            d = dists[i, j]
            if d > 1e-10:
                grad[i] += w * (-0.5) * (pts[i] - pts[j]) / d
    
    # Objective is -soft_min, so gradient is -grad_soft_min
    # grad_soft_min w.r.t. params is what we computed as 'grad' above
    # So objective gradient = -grad
    obj_val = -soft_min_val
    grad_obj = -grad
    
    return obj_val, grad_obj.flatten()


def _force_relax(pts, n_iters=2000, lr=0.003, k_repel=1.0):
    """
    Force-directed relaxation using cKDTree for neighbor lookup.
    """
    pts = pts.copy()
    n = len(pts)
    
    for it in range(n_iters):
        if _remaining() < 15.0:
            break
        
        # Build KDTree
        tree = cKDTree(pts)
        
        # Get neighbors within a cutoff
        cutoff = 0.2
        neighbors = tree.query_ball_point(pts, r=cutoff)
        
        forces = np.zeros_like(pts)
        
        for i in range(n):
            for j in neighbors[i]:
                if j <= i:
                    continue
                d = pts[i] - pts[j]
                dist = np.linalg.norm(d)
                if dist < 1e-10:
                    continue
                # Repulsive force ~ 1/dist^2
                # Only if dist < some threshold
                if dist < 0.15:
                    force_mag = k_repel / (dist * dist + 1e-8)
                    forces[i] += force_mag * d / dist
                    forces[j] -= force_mag * d / dist
        
        # Boundary: push inward if too close to edge
        dist_to_center = np.linalg.norm(pts, axis=1)
        r_est = _compute_radius(pts)
        margin = 0.05 * r_est + 1e-4
        for i in range(n):
            if dist_to_center[i] > 1.0 - margin:
                direction = -pts[i] / (dist_to_center[i] + 1e-10)
                strength = 0.5 * (dist_to_center[i] - (1.0 - margin))
                forces[i] += strength * direction
        
        # Update
        pts += lr * forces
        
        # Keep inside disk
        dists = np.linalg.norm(pts, axis=1)
        mask = dists > 1.0
        if np.any(mask):
            pts[mask] /= dists[mask, np.newaxis] * (1.0 + 1e-8)
        
        # Decay learning rate
        lr *= 0.9995
    
    return pts


def _local_polish(pts, max_time=20.0):
    """
    Polish using L-BFGS-B with the soft-min objective.
    """
    start = time.time()
    params = pts.flatten()
    
    def obj(p):
        if time.time() - start > max_time:
            return 0.0
        return _objective_and_grad(p, k_soft=50.0)[0]
    
    def grad(p):
        if time.time() - start > max_time:
            return np.zeros_like(p)
        return _objective_and_grad(p, k_soft=50.0)[1]
    
    result = minimize(obj, params, jac=grad, method='L-BFGS-B',
                      options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8,
                               'maxfun': 5000})
    
    if result.success or result.fun < 0:
        pts_polished = result.x.reshape(-1, 2)
        # Ensure inside disk
        dists = np.linalg.norm(pts_polished, axis=1)
        mask = dists > 1.0
        if np.any(mask):
            pts_polished[mask] /= dists[mask, np.newaxis] * (1.0 + 1e-8)
        return pts_polished
    
    return pts


def _anneal_restart(pts, n_restarts=3, max_time_per_restart=10.0):
    """
    Simulated annealing restarts to escape local optima.
    """
    best_pts = pts.copy()
    best_radius = _compute_radius(pts)
    
    for r in range(n_restarts):
        if _remaining() < max_time_per_restart + 5.0:
            break
        
        start = time.time()
        # Perturb the current best
        pts_perturbed = best_pts + np.random.normal(0, 0.02, size=best_pts.shape)
        # Keep inside disk
        dists = np.linalg.norm(pts_perturbed, axis=1)
        mask = dists > 1.0
        if np.any(mask):
            pts_perturbed[mask] /= dists[mask, np.newaxis] * (1.0 + 1e-8)
        
        # Force relax
        pts_relaxed = _force_relax(pts_perturbed, n_iters=500, lr=0.002, k_repel=1.0)
        
        # Local polish
        pts_polished = _local_polish(pts_relaxed, max_time=min(5.0, max_time_per_restart - 2.0))
        
        r_new = _compute_radius(pts_polished)
        if r_new > best_radius:
            best_radius = r_new
            best_pts = pts_polished.copy()
    
    return best_pts


def construct():
    """
    Construct the optimal packing of 300 circles in the unit disk.
    """
    best_pts = None
    best_radius = 0.0
    
    # Try multiple seeds
    seeds = [42, 123, 456, 789, 1011, 2022]
    
    for seed in seeds:
        if _remaining() < 25.0:
            break
        
        # Generate initial configuration
        pts = _hex_init(seed)
        
        # Force relaxation
        pts = _force_relax(pts, n_iters=1500, lr=0.004, k_repel=1.0)
        
        # Local polish
        pts = _local_polish(pts, max_time=min(15.0, _remaining() - 15.0))
        
        # Evaluate
        r = _compute_radius(pts)
        
        if r > best_radius:
            best_radius = r
            best_pts = pts.copy()
    
    # If we have time, do anneal restarts on the best
    if best_pts is not None and _remaining() > 20.0:
        n_restarts = min(3, int((_remaining() - 10.0) / 12.0))
        if n_restarts > 0:
            best_pts = _anneal_restart(best_pts, n_restarts=n_restarts,
                                       max_time_per_restart=min(10.0, (_remaining() - 10.0) / n_restarts))
            best_radius = _compute_radius(best_pts)
    
    # Final polish if time permits
    if best_pts is not None and _remaining() > 10.0:
        best_pts = _local_polish(best_pts, max_time=min(10.0, _remaining() - 5.0))
        best_radius = _compute_radius(best_pts)
    
    # Ensure valid
    if best_pts is None:
        best_pts = _hex_init(42)
    
    # Final projection
    dists = np.linalg.norm(best_pts, axis=1)
    mask = dists > 1.0
    if np.any(mask):
        best_pts[mask] /= dists[mask, np.newaxis] * (1.0 + 1e-8)
    
    return best_pts


# Test
if __name__ == "__main__":
    pts = construct()
    r = _compute_radius(pts)
    print(f"Radius: {r}")
    print(f"Shape: {pts.shape}")
