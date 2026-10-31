"""Optimized 16-point configuration minimizing squared max/min distance ratio."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint


def _pairwise_sq_dists(pts):
    """Compute squared pairwise distances for n points."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sum(diff * diff, axis=-1)


def _score(pts):
    """Compute (max_dist / min_dist)^2. Returns inf if invalid."""
    if len(pts) != N:
        return np.inf
    # Check distinctness
    d2 = _pairwise_sq_dists(pts)
    np.fill_diagonal(d2, np.inf)
    min_d2 = np.min(d2)
    max_d2 = np.max(d2)
    if min_d2 <= 1e-15:
        return np.inf
    return max_d2 / min_d2


def _log_score(pts):
    """Log of the ratio for smoother optimization. Returns large value if invalid."""
    if len(pts) != N:
        return 1e10
    d2 = _pairwise_sq_dists(pts)
    np.fill_diagonal(d2, np.inf)
    min_d2 = np.min(d2)
    max_d2 = np.max(d2)
    if min_d2 <= 1e-15:
        return 1e10
    return np.log(max_d2 / min_d2)


def _objective_and_grad(x):
    """Objective: log(max_d2/min_d2). We want to minimize this."""
    pts = x.reshape(-1, 2)
    d2 = _pairwise_sq_dists(pts)
    np.fill_diagonal(d2, np.inf)
    
    min_d2 = np.min(d2)
    max_d2 = np.max(d2)
    
    if min_d2 <= 1e-15:
        return 1e10, np.zeros_like(x)
    
    val = np.log(max_d2 / min_d2)
    
    # Gradient computation via finite differences is too slow.
    # Use analytic gradient for log ratio.
    # Let f = log(max_d2) - log(min_d2)
    # d f / d x_i = d log(max_d2)/dx_i - d log(min_d2)/dx_i
    
    # Identify the pair for max and min
    idx_max = np.unravel_index(np.argmin(d2 + np.diag(np.inf * np.ones(N))), (N, N))
    # Actually argmin on flattened with diagonal inf gives the min pair.
    # Let's find indices properly.
    flat_d2 = d2.copy()
    np.fill_diagonal(flat_d2, np.inf)
    idx_min_flat = np.argmin(flat_d2)
    idx_max_flat = np.argmax(flat_d2)
    
    i_min, j_min = np.unravel_index(idx_min_flat, (N, N))
    i_max, j_max = np.unravel_index(idx_max_flat, (N, N))
    
    # d log(d2_ij) / d x_k = 2 * (d2_ij / d2_ij) * (x_k - x_j) ... wait
    # d2_ij = ||x_i - x_j||^2
    # d d2_ij / d x_k = 2 (x_k - x_j) if k == i, -2 (x_k - x_i) if k == j, 0 otherwise
    # d log(d2_ij) / d x_k = (1/d2_ij) * d d2_ij / d x_k
    
    grad = np.zeros_like(x)
    
    # Contribution from min term (subtracted)
    diff_min = pts[i_min] - pts[j_min]
    # d log(min_d2) / d x = 2 * (x_i - x_j) / min_d2 for i, -2*(x_i-x_j)/min_d2 for j
    # Vectorized:
    grad[i_min*2:(i_min+1)*2] -= 2 * diff_min / min_d2
    grad[j_min*2:(j_min+1)*2] += 2 * diff_min / min_d2
    
    # Contribution from max term (added)
    diff_max = pts[i_max] - pts[j_max]
    grad[i_max*2:(i_max+1)*2] += 2 * diff_max / max_d2
    grad[j_max*2:(j_max+1)*2] -= 2 * diff_max / max_d2
    
    return val, grad


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_pts = None
    best_score = np.inf
    
    # Initial guess: regular 4x4 grid
    side = int(np.ceil(N ** (1 / D)))
    grid_x, grid_y = np.meshgrid(range(side), range(side))
    init_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])[:N]
    init_pts = init_pts.astype(float)
    
    # Normalize initial points to unit scale for better conditioning
    center = np.mean(init_pts, axis=0)
    init_pts = init_pts - center
    scale = np.max(np.linalg.norm(init_pts, axis=1))
    if scale > 0:
        init_pts = init_pts / scale
    
    x0 = init_pts.ravel()
    
    # Use L-BFGS-B for unconstrained optimization of log ratio
    # We will run multiple restarts if time permits
    
    restarts = 0
    max_restarts = 10
    
    while time.time() < deadline and restarts < max_restarts:
        try:
            res = minimize(
                _objective_and_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            pts = res.x.reshape(-1, 2)
            score = _score(pts)
            
            if score < best_score:
                best_score = score
                best_pts = pts.copy()
            
            # Check if converged to a good solution
            if best_score < 1.05:
                break
            
            # Perturb for next restart
            if restarts < max_restarts - 1:
                # Add noise to escape local minima
                noise_scale = 0.1 / (restarts + 1)
                x0 = best_pts.ravel() + np.random.normal(0, noise_scale, size=N*2)
            else:
                break
                
        except Exception:
            pass
        
        restarts += 1
    
    if best_pts is None:
        return init_pts
    
    return best_pts
# EVOLVE-BLOCK-END
