"""Pack 50 equal spheres in a unit cube to maximize radius."""
import numpy as np
import time

N = 50
DEADLINE = 110.0  # seconds, safety margin from 120s limit

def _compute_radius(pts):
    """Compute the maximum common radius for given centers."""
    # Wall distances: min over all points and all 6 faces
    # Cube is [-0.5, 0.5]^3
    # min distance from any point to any wall
    # min_i min_k (0.5 - |c_ik|) is equivalent to min( min_i(0.5 - c_i0), min_i(0.5 + c_i0), ... )
    neg_walls = np.min(pts, axis=0) + 0.5
    pos_walls = np.min(-pts, axis=0) + 0.5
    wall_dists = np.min(np.minimum(neg_walls, pos_walls))
    # Pairwise distances
    from scipy.spatial.distance import pdist
    min_pair = pdist(pts, 'euclidean').min()
    return min(wall_dists, min_pair / 2.0)

def _focal_loss(pts, alpha=100.0):
    """Soft objective: maximize the minimum distance using focal loss."""
    from scipy.spatial.distance import pdist
    pair_dists = pdist(pts, 'euclidean')
    neg_walls = np.min(pts, axis=0) + 0.5
    pos_walls = np.min(-pts, axis=0) + 0.5
    wall_dists = np.min(np.minimum(neg_walls, pos_walls))
    # Combine: we want all pair distances and wall distances large
    # Use a soft minimum via focal loss
    all_dists = np.concatenate([pair_dists, [wall_dists * 2.0]])
    # Focal loss: -log(sum(exp(-alpha * d))) approximates -min(d) for large alpha
    # But more stable: use softmax-based
    min_d = all_dists.min()
    # Penalize distances that are close to the minimum
    exp_d = np.exp(-alpha * (all_dists - min_d))
    return -min_d - np.log(exp_d.sum()) / alpha

def _repulsion_force(pts, k=10.0):
    """Compute repulsion forces to push points apart."""
    n = len(pts)
    forces = np.zeros_like(pts)
    # Pairwise repulsion
    for i in range(n):
        for j in range(i+1, n):
            d = pts[i] - pts[j]
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                d = np.random.randn(3) * 1e-10
                dist = np.linalg.norm(d)
            # Strong repulsion when close
            f = k / (dist**2 + 1e-6)
            forces[i] += f * d / dist
            forces[j] -= f * d / dist
    # Wall repulsion
    for i in range(n):
        for axis in range(3):
            # Left wall
            dist_l = pts[i, axis] + 0.5
            if dist_l < 0.1:
                forces[i, axis] += k / (dist_l + 1e-6)
            # Right wall
            dist_r = 0.5 - pts[i, axis]
            if dist_r < 0.1:
                forces[i, axis] -= k / (dist_r + 1e-6)
    return forces

def _initialize_fcc():
    """Initialize with FCC-like lattice points."""
    # FCC has 4 points per unit cell
    # In a cube of side 1, we can fit roughly 5x5x5 = 125 grid positions
    # But FCC is denser. Let's use a simpler approach:
    # Start with a 4x4x4 grid = 64 points, take first 50
    # Or better: use a body-centered cubic arrangement
    
    # Actually, let's use a well-spaced initial configuration
    # 50 points: try a 5x5x2 arrangement with jitter
    pts = []
    # Use a 5x5x2 grid with FCC-like offsets
    for i in range(5):
        for j in range(5):
            for k in range(2):
                x = (i + 0.5) / 5.0 - 0.5
                y = (j + 0.5) / 5.0 - 0.5
                z = (k + 0.5) / 2.0 - 0.5
                pts.append([x, y, z])
                if len(pts) >= N:
                    break
            if len(pts) >= N:
                break
        if len(pts) >= N:
            break
    pts = np.array(pts[:N])
    # Add small random jitter to break symmetry
    rng = np.random.default_rng(42)
    pts += rng.uniform(-0.01, 0.01, size=pts.shape)
    # Clip to cube
    pts = np.clip(pts, -0.49, 0.49)
    return pts

def _optimize_repulsion(pts, time_budget=60.0):
    """Optimize using repulsion-based iterative method."""
    start_time = time.time()
    current_pts = pts.copy()
    current_r = _compute_radius(current_pts)
    best_pts = current_pts.copy()
    best_r = current_r
    
    lr = 0.005
    k = 5.0
    
    while time.time() - start_time < time_budget:
        forces = _repulsion_force(current_pts, k)
        # Normalize forces
        force_mag = np.linalg.norm(forces, axis=1, keepdims=True)
        force_mag = np.maximum(force_mag, 1e-10)
        forces = forces / force_mag
        
        # Update
        new_pts = current_pts + lr * forces
        # Clip to cube
        new_pts = np.clip(new_pts, -0.499, 0.499)
        
        new_r = _compute_radius(new_pts)
        if new_r > current_r:
            current_pts = new_pts
            current_r = new_r
            if current_r > best_r:
                best_pts = current_pts.copy()
                best_r = current_r
            # Decrease learning rate slowly
            lr *= 0.999
        else:
            # Try smaller step
            lr *= 0.95
            if lr < 1e-6:
                # Restart with fresh random perturbation
                rng = np.random.default_rng()
                perturbed = current_pts + rng.normal(0, 0.01, size=current_pts.shape)
                perturbed = np.clip(perturbed, -0.499, 0.499)
                pert_r = _compute_radius(perturbed)
                if pert_r > best_r:
                    best_pts = perturbed.copy()
                    best_r = pert_r
                current_pts = perturbed
                current_r = pert_r
                lr = 0.005
                k = 5.0
    
    return best_pts, best_r

def _optimize_scipy(pts, time_budget=30.0):
    """Optimize using scipy's L-BFGS-B with a smooth objective."""
    from scipy.optimize import minimize
    start_time = time.time()
    
    def objective(x):
        if time.time() - start_time > time_budget:
            return 1e10  # Penalize to stop
        p = x.reshape(-1, 3)
        # Soft minimum objective
        from scipy.spatial.distance import pdist
        pair_dists = pdist(p, 'euclidean')
        neg_walls = np.min(p, axis=0) + 0.5
        pos_walls = np.min(-p, axis=0) + 0.5
        wall_dists = np.min(np.minimum(neg_walls, pos_walls))
        # We want to maximize min(pair_dists/2, wall_dists)
        # Use a smooth approximation: -log-sum-exp of negative distances
        alpha = 50.0
        all_dists = np.concatenate([pair_dists, [wall_dists * 2.0]])
        # Soft minimum
        min_d = all_dists.min()
        exp_vals = np.exp(-alpha * (all_dists - min_d))
        soft_min = min_d - np.log(exp_vals.sum()) / alpha
        return -soft_min  # Minimize negative
    
    x0 = pts.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
        if result.success or result.fun < objective(x0):
            new_pts = result.x.reshape(-1, 3)
            new_r = _compute_radius(new_pts)
            if new_r > _compute_radius(pts):
                return new_pts
    except Exception:
        pass
    
    return pts

def construct():
    start_time = time.time()
    
    # Phase 1: Initialize with FCC-like grid
    pts = _initialize_fcc()
    r = _compute_radius(pts)
    
    # Phase 2: Repulsion-based optimization (main workhorse)
    time_remaining = DEADLINE - (time.time() - start_time) - 5.0  # Reserve 5s
    if time_remaining > 10:
        pts, r = _optimize_repulsion(pts, time_budget=min(time_remaining, 80.0))
    
    # Phase 3: Refine with scipy if time permits
    time_remaining = DEADLINE - (time.time() - start_time) - 2.0
    if time_remaining > 10:
        pts = _optimize_scipy(pts, time_budget=min(time_remaining, 25.0))
        r = _compute_radius(pts)
    
    # Ensure validity: clip to cube
    pts = np.clip(pts, -0.5, 0.5)
    
    # Final check: if any issues, return safe grid
    if not np.all(np.isfinite(pts)) or pts.shape != (N, 3):
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
    
    return pts
