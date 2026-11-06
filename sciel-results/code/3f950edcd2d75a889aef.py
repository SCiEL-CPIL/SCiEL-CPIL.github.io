"""Packing 300 equal circles in a unit circle.

Strategy:
1. Generate a hexagonal lattice with fine spacing; pick the N closest-to-origin
   points that maximise the achievable radius.
2. Refine with a vectorised local-search / repulsion loop:
   - Each iteration picks a random point, proposes a small perturbation,
     and accepts if the computed radius improves.
   - Also run a repulsion-based relaxation pass to untangle close pairs.
3. Deadline guard ensures we return the best configuration found before
   the 150 s wall-clock kill.
"""
import numpy as np
import time

N = 300

def _compute_radius(pts):
    """Compute the radius for a configuration of N points."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Boundary constraint
    dists_origin = np.hypot(pts[:, 0], pts[:, 1])
    boundary_min = 1.0 - np.max(dists_origin)
    if boundary_min <= 0:
        return 0.0
    # Pairwise distances using vectorized approach
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    return min(min_pair / 2.0, boundary_min)


def _compute_radius_fast(pts):
    """Faster radius computation using only nearest-neighbor info."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    dists_origin = np.hypot(pts[:, 0], pts[:, 1])
    boundary_min = 1.0 - np.max(dists_origin)
    if boundary_min <= 0:
        return 0.0
    # Use scipy's cKDTree for efficient nearest neighbor
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    # Get 2 nearest neighbors for each point (one is self, so we need 2)
    dists, _ = tree.query(pts, k=2)
    min_pair = np.min(dists[:, 1])  # exclude self-distance
    return min(min_pair / 2.0, boundary_min)


def _hex_initial(spacing, n_points=N):
    """Generate hexagonal lattice points and select the best n_points."""
    sqrt3 = np.sqrt(3.0)
    k = int(1.0 / spacing) + 3
    
    points_list = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * sqrt3 / 2.0
            r = np.hypot(x, y)
            if r < 1.0:
                points_list.append((x, y))
    
    if len(points_list) < n_points:
        return None, 0.0
    
    points = np.array(points_list)
    dists = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(dists, kind='stable')[:n_points]
    pts = points[idx]
    r = _compute_radius(pts)
    return pts, r


def _local_search(pts, deadline, seed=42, use_fast=True):
    """Local search: perturb random points, accept if radius improves."""
    rng = np.random.RandomState(seed)
    current = pts.copy()
    compute_r = _compute_radius_fast if use_fast else _compute_radius
    current_r = compute_r(current)
    best = current.copy()
    best_r = current_r
    
    step_size = 0.005
    patience = 0
    max_patience = 300
    
    while time.time() < deadline and patience < max_patience:
        idx = rng.randint(N)
        old_pos = current[idx].copy()
        
        # Propose perturbation
        new_pos = old_pos + rng.uniform(-step_size, step_size, 2)
        
        # Keep inside circle
        new_r_mag = np.hypot(new_pos[0], new_pos[1])
        if new_r_mag > 0.99:
            new_pos = new_pos / new_r_mag * 0.99
        
        current[idx] = new_pos
        new_r = compute_r(current)
        
        if new_r > current_r + 1e-12:
            current_r = new_r
            patience = 0
            if new_r > best_r:
                best_r = new_r
                best = current.copy()
        else:
            current[idx] = old_pos
            patience += 1
            # Adaptive step size
            if patience % 50 == 0:
                step_size *= 0.95
            if step_size < 1e-5:
                break
    
    return best, best_r


def _repulsion_relax(pts, deadline, seed=123):
    """Repulsion-based relaxation to untangle close pairs."""
    rng = np.random.RandomState(seed)
    current = pts.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    # Iterative repulsion: push apart points that are too close
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        # Find the closest pair using KDTree
        from scipy.spatial import cKDTree
        tree = cKDTree(current)
        dists, pairs = tree.query(current, k=2)
        
        min_dist = np.min(dists[:, 1])
        if min_dist > 0.107:  # Above target spacing
            break
        
        # Find indices of closest pair
        min_idx = np.argmin(dists[:, 1])
        i, j = pairs[min_idx, 0], pairs[min_idx, 1]
        
        # Push them apart
        direction = current[i] - current[j]
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            direction = rng.randn(2)
            norm = np.linalg.norm(direction)
        direction = direction / norm
        
        push = 0.003
        current[i] += direction * push
        current[j] -= direction * push
        
        # Keep inside circle
        for k in [i, j]:
            r_mag = np.hypot(current[k, 0], current[k, 1])
            if r_mag > 0.995:
                current[k] = current[k] / r_mag * 0.995
        
        new_r = _compute_radius(current)
        if new_r > current_r + 1e-12:
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best = current.copy()
    
    return best, best_r


def _gradient_ascent(pts, deadline, seed=456):
    """Gradient ascent on a smooth surrogate objective."""
    from scipy.optimize import minimize
    
    rng = np.random.RandomState(seed)
    current = pts.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    # Define smooth surrogate using log-sum-exp
    def objective(x):
        x = x.reshape(-1, 2)
        # Boundary term
        dists_origin = np.hypot(x[:, 0], x[:, 1])
        boundary_term = np.max(dists_origin)
        
        # Pairwise distances - use soft min via log-sum-exp
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        
        # Soft min of distances
        beta = 50.0
        log_sum_exp = np.log(np.sum(np.exp(-beta * dists_sq)))
        soft_min_dist = -np.log(np.sum(np.exp(-beta * dists_sq))) / beta
        
        # Objective: minimize negative radius
        # radius = min(min_pair/2, 1 - max_dist)
        neg_radius = -min(soft_min_dist / 2.0, 1.0 - boundary_term)
        return neg_radius
    
    def objective_grad(x):
        x = x.reshape(-1, 2)
        dists_origin = np.hypot(x[:, 0], x[:, 1])
        boundary_term = np.max(dists_origin)
        
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        
        beta = 50.0
        weights = np.exp(-beta * dists_sq)
        weights /= np.sum(weights)
        
        # Gradient of soft min
        grad = np.zeros_like(x)
        for i in range(N):
            for j in range(N):
                if i != j:
                    grad[i] += weights[i, j] * 2.0 * (x[i] - x[j])
        
        # Gradient of boundary
        max_idx = np.argmax(dists_origin)
        boundary_grad = np.zeros_like(x)
        boundary_grad[max_idx] = x[max_idx] / dists_origin[max_idx]
        
        # Combine
        soft_min_dist = -np.log(np.sum(np.exp(-beta * dists_sq))) / beta
        radius = min(soft_min_dist / 2.0, 1.0 - boundary_term)
        
        if soft_min_dist / 2.0 < 1.0 - boundary_term:
            total_grad = grad / 2.0
        else:
            total_grad = boundary_grad
        
        return total_grad.flatten()
    
    # Run a few iterations of gradient ascent
    x = current.flatten()
    for _ in range(100):
        if time.time() > deadline:
            break
        
        try:
            result = minimize(objective, x, jac=objective_grad, method='L-BFGS-B',
                             options={'maxiter': 50, 'gtol': 1e-6})
            x_new = result.x.reshape(-1, 2)
            
            # Check if improvement
            new_r = _compute_radius(x_new)
            if new_r > current_r + 1e-12:
                current = x_new
                current_r = new_r
                if new_r > best_r:
                    best_r = new_r
                    best = current.copy()
        except Exception:
            break
    
    return best, best_r


def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    def time_check():
        return time.time() < deadline
    
    # --- Stage 1: Construct a good initial configuration ---
    best_points = None
    best_radius = 0.0
    
    # Try multiple spacings
    for trial in range(30):
        if not time_check():
            break
        
        spacing = 0.085 + trial * 0.001
        pts, r = _hex_initial(spacing)
        
        if pts is not None and r > best_radius:
            best_radius = r
            best_points = pts.copy()
    
    if best_points is None:
        # Fallback: uniform random in circle
        rng = np.random.RandomState(0)
        angles = rng.uniform(0, 2 * np.pi, N)
        radii = np.sqrt(rng.uniform(0, 1, N))
        best_points = np.column_stack([
            radii * np.cos(angles),
            radii * np.sin(angles)
        ])
        best_radius = _compute_radius(best_points)
    
    # --- Stage 2: Local optimization ---
    # Run repulsion relaxation first
    rep_budget = deadline - 30.0
    rep_pts, rep_r = _repulsion_relax(best_points, rep_budget, seed=123)
    if rep_r > best_radius:
        best_points = rep_pts
        best_radius = rep_r
    
    # Run local search with different seeds
    for seed in [42, 777, 999]:
        if not time_check():
            break
        
        ls_budget = deadline - 15.0
        ls_pts, ls_r = _local_search(best_points, ls_budget, seed=seed, use_fast=True)
        if ls_r > best_radius:
            best_points = ls_pts
            best_radius = ls_r
    
    # Run gradient ascent if time permits
    if time_check():
        grad_budget = deadline - 5.0
        grad_pts, grad_r = _gradient_ascent(best_points, grad_budget, seed=456)
        if grad_r > best_radius:
            best_points = grad_pts
            best_radius = grad_r
    
    # Final high-precision polish
    if time_check():
        polish_budget = deadline - 2.0
        polish_pts, polish_r = _local_search(best_points, polish_budget, seed=12345, use_fast=False)
        if polish_r > best_radius:
            best_points = polish_pts
            best_radius = polish_r
    
    # Final validation and return
    # Ensure points are strictly inside
    dists_origin = np.hypot(best_points[:, 0], best_points[:, 1])
    max_allowed = 1.0 - 1e-10
    mask = dists_origin >= max_allowed
    if np.any(mask):
        best_points[mask] *= (max_allowed / dists_origin[mask]) * 0.999
    
    # Ensure no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(best_points[i, 0] - best_points[j, 0], best_points[i, 1] - best_points[j, 1])
            if d < 1e-6:
                angle = np.random.uniform(0, 2 * np.pi)
                best_points[i, 0] += 0.01 * np.cos(angle)
                best_points[i, 1] += 0.01 * np.sin(angle)
                best_points[j, 0] -= 0.01 * np.cos(angle)
                best_points[j, 1] -= 0.01 * np.sin(angle)
    
    # Final check
    if not np.all(np.isfinite(best_points)) or best_points.shape != (N, 2):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = np.sqrt(np.linspace(0, 1, N))
        best_points = np.column_stack([
            radii * np.cos(angles),
            radii * np.sin(angles)
        ])
        best_points *= 0.99
    
    return best_points
