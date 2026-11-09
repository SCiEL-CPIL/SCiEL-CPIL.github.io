"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
TIME_LIMIT = 110.0  # Safety margin for the 120s limit

# EVOLVE-BLOCK-START

def _radius(pts):
    """Compute the largest common radius for the given centres."""
    # Wall distance: min over i,k of (0.5 - |c_ik|)
    # c_ik is coordinate k of point i
    # dist to wall is 0.5 - max(|x|, |y|)
    wall_dists = 0.5 - np.max(np.abs(pts), axis=1)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    # Using broadcasting
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    
    return min(min_wall, min_pair)

def _penalty_smooth(pts_flat, r_target):
    """
    Objective function for L-BFGS-B.
    Minimizes the squared violation of the constraints for radius r_target.
    Constraints:
    1. d_ij >= 2 * r_target
    2. Points inside [-0.5 + r, 0.5 - r]^2
    """
    pts = pts_flat.reshape(N, 2)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(d, np.inf)
    
    # Violations for pairs
    # We want d >= 2*r_target
    viol_pairs = np.maximum(0, 2 * r_target - d)
    
    # Use triu_indices for efficiency in summing unique pairs
    i_idx, j_idx = np.triu_indices(N, k=1)
    viol_pairs_unique = viol_pairs[i_idx, j_idx]
    E_pairs = np.sum(viol_pairs_unique**2)
    
    # Wall distances
    # Instead of max(|x|, |y|), we use linear constraints:
    # x_i <= 0.5 - r, x_i >= -0.5 + r
    # y_i <= 0.5 - r, y_i >= -0.5 + r
    # This is smoother for optimization.
    
    x = pts[:, 0]
    y = pts[:, 1]
    
    # Violations for walls
    # Right wall: x > 0.5 - r
    viol_x_right = np.maximum(0, x - (0.5 - r_target))
    # Left wall: x < -0.5 + r
    viol_x_left = np.maximum(0, (-0.5 + r_target) - x)
    # Top wall: y > 0.5 - r
    viol_y_top = np.maximum(0, y - (0.5 - r_target))
    # Bottom wall: y < -0.5 + r
    viol_y_bottom = np.maximum(0, (-0.5 + r_target) - y)
    
    E_walls = np.sum(viol_x_right**2) + np.sum(viol_x_left**2) + \
              np.sum(viol_y_top**2) + np.sum(viol_y_bottom**2)
    
    return E_pairs + E_walls

def _grad_penalty_smooth(pts_flat, r_target):
    """
    Gradient of the smooth penalty function.
    """
    pts = pts_flat.reshape(N, 2)
    grad = np.zeros_like(pts)
    
    # Pairwise gradient
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(d, np.inf)
    
    # Avoid division by zero
    d_safe = np.where(d > 1e-8, d, 1.0)
    
    # viol = max(0, 2r - d)
    # d(viol^2)/d(pts_i) = 2 * viol * d(viol)/d(pts_i)
    # d(viol)/d(pts_i) = - (pts_i - pts_j) / d_ij  if viol > 0
    # So d(E)/d(pts_i) = sum_j 2 * viol_ij * (-(pts_i - pts_j) / d_ij)
    
    # We only care about viol > 0
    mask = (2 * r_target - d) > 0
    # Set diagonal to False just in case
    np.fill_diagonal(mask, False)
    
    # Weight matrix W_ij = 2 * viol_ij / d_ij
    W = np.zeros((N, N))
    W[mask] = 2 * (2 * r_target - d[mask]) / d_safe[mask]
    
    # grad_i = - sum_j W_ij (pts_i - pts_j) = - (sum_j W_ij) pts_i + sum_j W_ij pts_j
    row_sums = W.sum(axis=1)
    grad_pairs = -row_sums[:, np.newaxis] * pts + W.dot(pts)
    
    # Wall gradient
    # E_walls = sum (max(0, x - (0.5-r))^2 + max(0, (-0.5+r) - x)^2 + ...)
    # d/dx (max(0, x - a)^2) = 2 * max(0, x - a)
    # d/dx (max(0, b - x)^2) = -2 * max(0, b - x)
    
    x = pts[:, 0]
    y = pts[:, 1]
    
    viol_x_right = np.maximum(0, x - (0.5 - r_target))
    viol_x_left = np.maximum(0, (-0.5 + r_target) - x)
    viol_y_top = np.maximum(0, y - (0.5 - r_target))
    viol_y_bottom = np.maximum(0, (-0.5 + r_target) - y)
    
    grad_walls = np.zeros_like(pts)
    grad_walls[:, 0] = 2 * viol_x_right - 2 * viol_x_left
    grad_walls[:, 1] = 2 * viol_y_top - 2 * viol_y_bottom
    
    grad = grad_pairs + grad_walls
    return grad.flatten()

def _init_grid(jitter=0.0):
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])[:N]
    if jitter > 0:
        rng = np.random.default_rng(42)
        grid = grid + rng.normal(0, jitter, grid.shape)
        grid = np.clip(grid, -0.49, 0.49)
    return grid

def construct():
    t0 = time.time()
    
    # Initial guess: Grid
    best_pts = _init_grid()
    best_r = _radius(best_pts)
    
    # Binary search for radius
    r_low = best_r
    r_high = 0.08
    
    rng = np.random.default_rng(42)
    
    # We will try to improve r_low
    # Iterate binary search steps
    max_steps = 15
    for step in range(max_steps):
        if time.time() - t0 > TIME_LIMIT - 20:
            break
            
        r_mid = (r_low + r_high) / 2.0
        
        # Try to find a feasible configuration for r_mid
        feasible = False
        best_pts_for_r = None
        best_r_for_r = 0
        
        # Multiple restarts to avoid local minima
        # We use different initialization strategies
        n_restarts = 4
        for k in range(n_restarts):
            if time.time() - t0 > TIME_LIMIT - 15:
                break
                
            if k == 0:
                init_pts = _init_grid(jitter=0.005)
            elif k == 1:
                init_pts = _init_grid(jitter=0.01)
            elif k == 2:
                # Random start with repulsion
                init_pts = rng.uniform(-0.45, 0.45, size=(N, 2))
                # Simple push apart
                for _ in range(10):
                    d = np.sqrt(np.sum((init_pts[:, None, :] - init_pts[None, :, :]) ** 2, axis=2))
                    np.fill_diagonal(d, np.inf)
                    for i in range(N):
                        for j in range(i + 1, N):
                            if d[i, j] < 0.05:
                                diff = init_pts[i] - init_pts[j]
                                if np.linalg.norm(diff) < 1e-8:
                                    diff = rng.normal(size=2)
                                diff /= np.linalg.norm(diff)
                                init_pts[i] += 0.5 * diff * 0.05
                                init_pts[j] -= 0.5 * diff * 0.05
            else:
                # Jittered previous best if available
                if best_pts_for_r is not None:
                    init_pts = best_pts_for_r + rng.normal(0, 0.002, best_pts_for_r.shape)
                else:
                    init_pts = _init_grid(jitter=0.02)
            
            init_pts = np.clip(init_pts, -0.5, 0.5)
            
            try:
                res = minimize(
                    _penalty_smooth,
                    init_pts.flatten(),
                    args=(r_mid,),
                    jac=_grad_penalty_smooth,
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * (2 * N),
                    options={'maxiter': 300, 'gtol': 1e-5}
                )
                
                # Check if penalty is close to 0
                if res.fun < 1e-6:
                    cand_pts = res.x.reshape(N, 2)
                    # Verify radius
                    r_cand = _radius(cand_pts)
                    if r_cand >= r_mid - 1e-5:
                        feasible = True
                        if r_cand > best_r_for_r:
                            best_r_for_r = r_cand
                            best_pts_for_r = cand_pts
            except Exception:
                pass
        
        if feasible and best_pts_for_r is not None:
            r_low = r_mid
            best_pts = best_pts_for_r
            best_r = best_r_for_r
        else:
            r_high = r_mid
            
        # If we've converged enough
        if r_high - r_low < 1e-6:
            break

    # Final polish: Try to increase radius slightly from the best found
    # This handles the case where binary search stopped but we might squeeze a bit more
    # by using the last feasible point as a starting point for a slightly higher r
    if time.time() - t0 < TIME_LIMIT - 5:
        # Try a few small increases
        current_r = _radius(best_pts)
        for _ in range(3):
            r_try = current_r * 1.001 + 1e-6
            init_pts = best_pts + rng.normal(0, 0.001, best_pts.shape)
            try:
                res = minimize(
                    _penalty_smooth,
                    init_pts.flatten(),
                    args=(r_try,),
                    jac=_grad_penalty_smooth,
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * (2 * N),
                    options={'maxiter': 100, 'gtol': 1e-5}
                )
                if res.fun < 1e-6:
                    cand_pts = res.x.reshape(N, 2)
                    r_cand = _radius(cand_pts)
                    if r_cand > current_r:
                        best_pts = cand_pts
                        current_r = r_cand
                    else:
                        break
                else:
                    break
            except Exception:
                break
        best_r = _radius(best_pts)

    return best_pts
# EVOLVE-BLOCK-END
