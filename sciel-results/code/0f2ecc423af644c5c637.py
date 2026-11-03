"""
Pack 200 equal circles in a unit square using a combination of:
1. Gradient-based optimization with soft-min objective (from current program)
2. Force-directed repulsion with KDTree acceleration (from inspiration)
3. Multi-start with diverse initializations (hex grids, random, grid)
4. Final high-precision polish pass

The key insight: use force-directed to quickly spread out points, then use
gradient-based soft-min optimization for fine-tuning.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

N = 200
_TIME_LIMIT = 135  # Leave margin for 150s limit


def _get_radius(pts):
    """Calculate the maximum possible radius for given centers."""
    if len(pts) == 0:
        return 0.0
    
    # Wall distances
    min_x = np.min(pts[:, 0])
    max_x = np.max(pts[:, 0])
    min_y = np.min(pts[:, 1])
    max_y = np.max(pts[:, 1])
    
    wall_dist = min(min_x + 0.5, 0.5 - max_x, min_y + 0.5, 0.5 - max_y)
    
    # Pairwise distances using cdist (fast for N=200)
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D) / 2.0
    
    return min(wall_dist, min_pair)


def _soft_min(vals, beta):
    """Compute the soft minimum of a set of values."""
    y = -beta * vals
    max_y = np.max(y)
    if not np.isfinite(max_y):
        return np.min(vals)
    shifted = y - max_y
    sum_exp = np.sum(np.exp(shifted))
    if sum_exp == 0:
        return np.min(vals)
    return -(max_y + np.log(sum_exp)) / beta


def _objective_and_grad(x, beta):
    """
    Objective function and gradient for L-BFGS-B.
    We want to MAXIMIZE the soft-min of constraints, so MINIMIZE -soft_min.
    """
    pts = x.reshape((N, 2))
    
    # Build constraint array C
    i_idx, j_idx = np.triu_indices(N, k=1)
    num_pairs = len(i_idx)
    
    C = np.empty(4*N + num_pairs)
    
    # Wall values
    C[0:N] = pts[:, 0] + 0.5
    C[N:2*N] = 0.5 - pts[:, 0]
    C[2*N:3*N] = pts[:, 1] + 0.5
    C[3*N:4*N] = 0.5 - pts[:, 1]
    
    # Pair values
    dx = pts[i_idx, 0] - pts[j_idx, 0]
    dy = pts[i_idx, 1] - pts[j_idx, 1]
    dists = np.sqrt(dx**2 + dy**2)
    C[4*N:] = dists / 2.0
    
    # Soft min computation
    y = -beta * C
    max_y = np.max(y)
    if not np.isfinite(max_y):
        min_val = np.min(C)
        return -min_val, np.zeros_like(x)
    
    shifted = y - max_y
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    if sum_exp == 0:
        min_val = np.min(C)
        return -min_val, np.zeros_like(x)
    
    soft_min_val = -(max_y + np.log(sum_exp)) / beta
    obj_val = -soft_min_val
    
    # Weights
    weights = exp_shifted / sum_exp
    
    # Gradient computation
    grad_pts = np.zeros((N, 2))
    
    # Wall gradients
    grad_pts[:, 0] += weights[0:N] - weights[N:2*N]
    grad_pts[:, 1] += weights[2*N:3*N] - weights[3*N:4*N]
    
    # Pair gradients
    valid = dists > 1e-10
    dx_safe = np.where(valid, dx, 1.0)
    dy_safe = np.where(valid, dy, 1.0)
    dists_safe = np.where(valid, dists, 1.0)
    
    grad_pair_x_i = np.where(valid, dx_safe / (2.0 * dists_safe), 0.0)
    grad_pair_x_j = np.where(valid, -dx_safe / (2.0 * dists_safe), 0.0)
    grad_pair_y_i = np.where(valid, dy_safe / (2.0 * dists_safe), 0.0)
    grad_pair_y_j = np.where(valid, -dy_safe / (2.0 * dists_safe), 0.0)
    
    pair_weights = weights[4*N:]
    
    np.add.at(grad_pts[:, 0], i_idx, grad_pair_x_i * pair_weights)
    np.add.at(grad_pts[:, 0], j_idx, grad_pair_x_j * pair_weights)
    np.add.at(grad_pts[:, 1], i_idx, grad_pair_y_i * pair_weights)
    np.add.at(grad_pts[:, 1], j_idx, grad_pair_y_j * pair_weights)
    
    grad_pts = -grad_pts / beta
    
    return obj_val, grad_pts.flatten()


def _force_directed_step(pts, r_target, lr=0.001):
    """
    One step of force-directed optimization.
    Uses KDTree for efficient neighbor queries.
    """
    n = len(pts)
    forces = np.zeros_like(pts)
    
    # Wall forces
    for i in range(n):
        if pts[i, 0] < -0.5 + r_target:
            forces[i, 0] += 5.0 * (r_target - (pts[i, 0] + 0.5))
        if pts[i, 0] > 0.5 - r_target:
            forces[i, 0] -= 5.0 * (pts[i, 0] - (0.5 - r_target))
        if pts[i, 1] < -0.5 + r_target:
            forces[i, 1] += 5.0 * (r_target - (pts[i, 1] + 0.5))
        if pts[i, 1] > 0.5 - r_target:
            forces[i, 1] -= 5.0 * (pts[i, 1] - (0.5 - r_target))
    
    # Repulsive forces using KDTree
    tree = KDTree(pts)
    cutoff = max(2 * r_target, 0.05)
    
    for i in range(n):
        neighbors = tree.query_ball_point(pts[i], cutoff)
        for j in neighbors:
            if i == j:
                continue
            diff = pts[i] - pts[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                forces[i] += np.random.randn(2) * 0.01
                continue
            if dist < 2 * r_target:
                overlap = 2 * r_target - dist
                force_mag = 10.0 * overlap / dist
                forces[i] += force_mag * diff
    
    pts += lr * forces
    pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
    pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)
    
    return pts


def _generate_hex_initial(spacing=0.078):
    """Generate a centered hexagonal lattice packing."""
    points = []
    row_h = spacing * np.sqrt(3) / 2
    
    y = -1.0
    row_idx = 0
    while y < 1.0:
        shift = (spacing / 2.0) if row_idx % 2 == 1 else 0.0
        x = -1.0 + shift
        while x < 1.0:
            points.append([x, y])
            x += spacing
        y += row_h
        row_idx += 1
        
    pts = np.array(points)
    
    # Filter points inside the square
    mask = (pts[:, 0] > -0.49) & (pts[:, 0] < 0.49) & (pts[:, 1] > -0.49) & (pts[:, 1] < 0.49)
    pts = pts[mask]
    
    if len(pts) < N:
        return np.random.uniform(-0.4, 0.4, (N, 2))
        
    # Select N points closest to the center
    dists_to_center = np.sum(pts**2, axis=1)
    indices = np.argsort(dists_to_center)
    pts = pts[indices[:N]]
    
    return pts


def _generate_grid_initial(rows=14, cols=15):
    """Generate a regular grid."""
    dx = 1.0 / cols
    dy = 1.0 / rows
    xs = -0.5 + (np.arange(cols) + 0.5) * dx
    ys = -0.5 + (np.arange(rows) + 0.5) * dy
    grid = np.array([(x, y) for y in ys for x in xs])
    
    if len(grid) >= N:
        # Select N points with best minimum distance (greedy)
        selected = []
        available = set(range(len(grid)))
        
        for _ in range(N):
            if not available:
                break
            best_idx = None
            best_min_dist = -1
            
            for idx in available:
                if not selected:
                    min_dist = np.inf
                else:
                    min_dist = np.min([np.linalg.norm(grid[idx] - grid[s]) for s in selected])
                
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_idx = idx
            
            selected.append(best_idx)
            available.discard(best_idx)
            
        return grid[selected]
    else:
        return np.random.uniform(-0.4, 0.4, (N, 2))


def _run_gradient_optimization(pts, time_limit, betas=None):
    """Run L-BFGS-B optimization with increasing beta."""
    if betas is None:
        betas = [5, 10, 20, 50, 100, 200]
        
    t_start = time.time()
    x0 = pts.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    best_r = _get_radius(pts)
    best_x = x0.copy()
    
    for beta in betas:
        if time.time() - t_start > time_limit:
            break
            
        def fun_and_grad(x):
            val, g = _objective_and_grad(x, beta)
            return val, g
            
        try:
            res = minimize(
                fun_and_grad, 
                x0, 
                jac=True, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            x0 = res.x
            pts_curr = x0.reshape((N, 2))
            pts_curr = np.clip(pts_curr, -0.5, 0.5)
            
            r = _get_radius(pts_curr)
            if r > best_r:
                best_r = r
                best_x = x0.copy()
                
        except Exception:
            pass
            
    return best_r, best_x.reshape((N, 2))


def _run_force_directed(pts, time_limit, max_iters=2000):
    """Run force-directed optimization."""
    t_start = time.time()
    pts = pts.copy()
    
    r_current = _get_radius(pts)
    lr = 0.001
    cooling = 0.999
    
    for it in range(max_iters):
        if time.time() - t_start > time_limit:
            break
        
        pts = _force_directed_step(pts, r_current, lr)
        lr *= cooling
        
        r_new = _get_radius(pts)
        if r_new > r_current:
            r_current = r_new
        
        if np.max(np.abs(pts - pts)) < 1e-8:
            break
    
    return r_current, pts


def _polish(pts, time_limit):
    """Final high-precision polish with high beta."""
    betas_polish = [500, 1000, 2000]
    t_start = time.time()
    x0 = pts.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    best_r = _get_radius(pts)
    best_x = x0.copy()
    
    for beta in betas_polish:
        if time.time() - t_start > time_limit:
            break
            
        def fun_and_grad(x):
            val, g = _objective_and_grad(x, beta)
            return val, g
            
        try:
            res = minimize(
                fun_and_grad, 
                x0, 
                jac=True, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            x0 = res.x
            pts_curr = x0.reshape((N, 2))
            pts_curr = np.clip(pts_curr, -0.5, 0.5)
            
            r = _get_radius(pts_curr)
            if r > best_r:
                best_r = r
                best_x = x0.copy()
                
        except Exception:
            pass
            
    return best_r, best_x.reshape((N, 2))


def construct():
    t0 = time.time()
    np.random.seed(42)
    
    best_r = 0.0
    best_pts = None
    
    def _update_best(r, pts):
        nonlocal best_r, best_pts
        if r > best_r:
            best_r = r
            best_pts = pts
    
    # Phase 1: Try multiple initializations with force-directed + gradient
    initializations = []
    
    # Hex grids with different spacings
    for spacing in [0.07, 0.075, 0.08, 0.085]:
        pts = _generate_hex_initial(spacing)
        initializations.append(pts)
    
    # Grid initialization
    pts = _generate_grid_initial(14, 15)
    initializations.append(pts)
    
    # Random initialization
    pts = np.random.uniform(-0.4, 0.4, (N, 2))
    initializations.append(pts)
    
    # Process each initialization
    for i, pts_init in enumerate(initializations):
        elapsed = time.time() - t0
        remaining = _TIME_LIMIT - elapsed - 10  # Leave 10s for polish
        
        if remaining <= 0:
            break
        
        # Split remaining time between force-directed and gradient
        fd_time = remaining * 0.3
        grad_time = remaining * 0.7
        
        # Force-directed phase
        if fd_time > 5:
            r_fd, pts_fd = _run_force_directed(pts_init, fd_time)
        else:
            r_fd, pts_fd = _get_radius(pts_init), pts_init
        
        _update_best(r_fd, pts_fd)
        
        # Gradient phase
        if grad_time > 5:
            r_grad, pts_grad = _run_gradient_optimization(pts_fd, grad_time)
            _update_best(r_grad, pts_grad)
    
    # Phase 2: Polish the best result
    if best_pts is not None:
        elapsed = time.time() - t0
        remaining = _TIME_LIMIT - elapsed - 2
        
        if remaining > 10:
            r_polish, pts_polish = _polish(best_pts, remaining)
            _update_best(r_polish, pts_polish)
    
    if best_pts is None:
        best_pts = np.random.uniform(-0.4, 0.4, (N, 2))
        
    return best_pts


if __name__ == "__main__":
    pts = construct()
    r = _get_radius(pts)
    print(f"Best Radius: {r}")
