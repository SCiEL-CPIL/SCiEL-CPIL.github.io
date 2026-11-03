"""Packing 200 equal circles in a unit square via smooth soft-min optimization."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _soft_min_radius(pts):
    """Compute a smooth surrogate for the minimum radius.
    
    r = min( min_{i<j} |c_i - c_j|/2, min_i min_k (1/2 - |c_ik|) )
    
    We maximize this. Use a log-sum-exp based soft-min for smoothness.
    """
    n = len(pts)
    # Pairwise distances
    # Use vectorized computation
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances: distance from each point to nearest wall
    wall_dists = np.array([
        0.5 - np.abs(pts[:, 0]),
        0.5 - np.abs(pts[:, 1])
    ])
    wall_min = np.min(wall_dists, axis=0)  # shape (n,)
    
    # Candidate radii: half pairwise distances and wall distances
    # We want the min over all these
    # Flatten pairwise distances (upper triangle only)
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu] / 2.0
    
    all_radii = np.concatenate([pair_dists, wall_min])
    
    # Soft min: -log-sum-exp(-x / beta) * beta
    beta = 0.001
    # Clip to avoid overflow
    vals = all_radii
    max_val = np.max(vals)
    log_sum = np.log(np.sum(np.exp(-(vals - max_val) / beta)))
    soft_min = max_val - beta * log_sum
    
    return soft_min


def _objective(pts):
    """Negative soft-min radius (to minimize)."""
    return -_soft_min_radius(pts)


def _grad_objective(pts):
    """Numerical gradient of the objective."""
    eps = 1e-7
    n = len(pts)
    grad = np.zeros_like(pts)
    base = _objective(pts)
    for i in range(n):
        for j in range(2):
            pts[i, j] += eps
            f_plus = _objective(pts)
            pts[i, j] -= 2 * eps
            f_minus = _objective(pts)
            pts[i, j] += eps
            grad[i, j] = (f_plus - f_minus) / (2 * eps)
    return grad


def _valid_solution(pts):
    """Check if solution is valid: all points in [-0.5, 0.5]^2, no coincident points."""
    if not np.all(np.isfinite(pts)):
        return False
    if np.any(pts < -0.5) or np.any(pts > 0.5):
        return False
    # Check for coincident points
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-12:
        return False
    return True


def _compute_radius(pts):
    """Compute the exact minimum radius for a valid configuration."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    wall_dists = np.array([
        0.5 - np.abs(pts[:, 0]),
        0.5 - np.abs(pts[:, 1])
    ])
    min_wall = np.min(wall_dists)
    
    return min(min_pair, min_wall)


def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    # Initial guess: grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    best_pts = grid[:N].copy()
    best_r = _compute_radius(best_pts)
    
    # Try multiple random restarts with L-BFGS-B
    np.random.seed(42)
    
    for restart in range(10):
        if time.time() - start_time > deadline:
            break
        
        # Initialize with slight perturbation of grid or random
        if restart == 0:
            pts = best_pts.copy()
            pts += np.random.normal(0, 0.001, pts.shape)
        else:
            # Random initialization within the square
            pts = np.random.uniform(-0.45, 0.45, (N, 2))
        
        # Clip to valid region
        pts = np.clip(pts, -0.49, 0.49)
        
        if not _valid_solution(pts):
            continue
        
        # Optimize
        try:
            result = minimize(
                _objective,
                pts,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 2000}
            )
        except Exception:
            continue
        
        opt_pts = result.x
        opt_pts = np.clip(opt_pts, -0.5, 0.5)
        
        if not _valid_solution(opt_pts):
            continue
        
        opt_r = _compute_radius(opt_pts)
        
        if opt_r > best_r:
            best_r = opt_r
            best_pts = opt_pts.copy()
    
    # Final polish: use tighter tolerance
    if time.time() - start_time < deadline - 5:
        try:
            result = minimize(
                _objective,
                best_pts,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 5000}
            )
            opt_pts = result.x
            opt_pts = np.clip(opt_pts, -0.5, 0.5)
            if _valid_solution(opt_pts):
                opt_r = _compute_radius(opt_pts)
                if opt_r > best_r:
                    best_r = opt_r
                    best_pts = opt_pts.copy()
        except Exception:
            pass
    
    # Ensure final validity
    if not _valid_solution(best_pts):
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_pts = grid[:N].copy()
    
    return best_pts
# EVOLVE-BLOCK-END
