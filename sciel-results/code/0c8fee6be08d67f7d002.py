"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def _compute_score(points):
    """Compute Delta / n^n after scaling to diameter 2."""
    n = len(points)
    # Compute all pairwise distances
    pts = np.array(points, dtype=np.float64)
    # Pairwise distance matrix
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    # Diameter = max distance
    dmax = np.max(dists)
    if dmax < 1e-15:
        return 0.0
    # Scale to diameter 2
    scaled_dists = dists / dmax * 2.0
    # Product over ordered pairs (i != j)
    # Use log for numerical stability
    mask = ~np.eye(n, dtype=bool)
    log_delta = np.sum(np.log(scaled_dists[mask] + 1e-300))
    score = math.exp(log_delta) / (n ** n)
    return score

def _objective(params):
    """Negative log of the score (we minimize this).
    
    params: 12 floats encoding 5 free points (the first 3 are fixed by gauge).
    Gauge: point 0 = (0,0), point 1 = (1,0), point 2 = (0, y2) with y2 > 0.
    Points 3-7 are free (10 coords) + y2 = 11... 
    Actually let's use a different parameterization.
    
    Better approach: use all 16 coords but fix gauge by:
    - Translate so centroid is at origin (removes 2 DOF)
    - Rotate so first point is on positive x-axis (removes 1 DOF)
    Total: 16 - 3 = 13 free params.
    
    Let's just optimize all 16 coordinates with the score being scale-invariant.
    The score is invariant under translation, rotation, and scaling, so we can
    just maximize the product of distances directly (since scaling to diameter 2
    just divides by dmax^56, and the product scales as dmax^56 for 56 pairs... 
    wait, 8 points -> 8*7 = 56 ordered pairs, so product scales as dmax^56).
    
    Actually the score = (prod of all ordered pair distances) / (dmax^56) / 8^8
    So maximizing score is equivalent to maximizing prod/dmax^56.
    
    Let's work with log_score = sum(log(d_ij)) - 56*log(dmax) - 8*log(8)
    and minimize -log_score.
    """
    pts = params.reshape(8, 2)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1) + 1e-30)
    dmax = np.max(dists)
    if dmax < 1e-15:
        return 1e10
    mask = ~np.eye(8, dtype=bool)
    log_prod = np.sum(np.log(dists[mask]))
    log_score = log_prod - 56.0 * np.log(dmax) - 8.0 * np.log(8.0)
    return -log_score

def _grad_objective(params):
    """Gradient of the objective (negative log score)."""
    pts = params.reshape(8, 2)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1) + 1e-30)
    dmax = np.max(dists)
    if dmax < 1e-15:
        return np.zeros(16)
    mask = ~np.eye(8, dtype=bool)
    
    # d(log d_ij)/d(pts_k) for k != i,j: 0
    # d(log d_ij)/d(pts_i) = (pts_i - pts_j) / d_ij^2
    # d(log d_ij)/d(pts_j) = (pts_j - pts_i) / d_ij^2
    
    grad = np.zeros_like(pts)
    for i in range(8):
        for j in range(8):
            if i == j:
                continue
            d2 = dists[i, j] ** 2
            if d2 < 1e-20:
                continue
            grad[i] += (pts[i] - pts[j]) / d2
            grad[j] += (pts[j] - pts[i]) / d2
    
    # d(-56*log(dmax))/d(pts)
    # Find the pair achieving dmax
    idx = np.unravel_index(np.argmax(dists), (8, 8))
    i_max, j_max = idx
    if i_max == j_max:
        return -grad  # shouldn't happen
    dmax_val = dists[i_max, j_max]
    # d(dmax)/d(pts_i) = (pts_i - pts_j) / dmax
    # d(-56 log dmax)/d(pts_i) = -56 * (pts_i - pts_j) / (dmax * dmax)
    # = -56 * (pts_i - pts_j) / dmax^2
    diff_max = pts[i_max] - pts[j_max]
    grad[i_max] -= 56.0 * diff_max / (dmax_val ** 2)
    grad[j_max] += 56.0 * diff_max / (dmax_val ** 2)
    
    # Objective is -log_score, so gradient of objective = -gradient of log_score
    # log_score = sum(log d_ij) - 56 log dmax - const
    # grad of log_score wrt pts is what we computed as 'grad'
    # So grad of objective = -grad
    return -grad.ravel()

def _regular_octagon_params():
    """Return params for regular octagon."""
    angles = 2 * np.pi * np.arange(8) / 8
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    return pts.ravel()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    from scipy.optimize import minimize
    
    best_params = _regular_octagon_params()
    best_score = _compute_score(best_params.reshape(8, 2))
    
    # Multi-restart optimization
    np.random.seed(42)
    
    # Generate diverse initial points
    def _make_initial(i):
        if i == 0:
            return _regular_octagon_params()
        elif i == 1:
            # Regular octagon with slight perturbation
            p = _regular_octagon_params().copy()
            p += 0.01 * np.random.randn(16)
            return p
        elif i == 2:
            # Two concentric squares
            r1, r2 = 1.0, 0.5
            angles1 = np.pi/4 + 2*np.pi*np.arange(4)/4
            angles2 = 2*np.pi*np.arange(4)/4
            pts = np.vstack([
                np.column_stack([r1*np.cos(angles1), r1*np.sin(angles1)]),
                np.column_stack([r2*np.cos(angles2), r2*np.sin(angles2)])
            ])
            return pts.ravel()
        elif i == 3:
            # Square with 4 extra points
            r1 = 1.0
            angles1 = 2*np.pi*np.arange(4)/4
            r2 = 0.7
            angles2 = np.pi/4 + 2*np.pi*np.arange(4)/4
            pts = np.vstack([
                np.column_stack([r1*np.cos(angles1), r1*np.sin(angles1)]),
                np.column_stack([r2*np.cos(angles2), r2*np.sin(angles2)])
            ])
            return pts.ravel()
        else:
            # Random perturbation of regular octagon
            p = _regular_octagon_params().copy()
            scale = 0.1 + 0.1 * i
            p += scale * np.random.randn(16)
            return p
    
    n_restarts = 20
    for restart in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        x0 = _make_initial(restart)
        
        try:
            result = minimize(
                _objective,
                x0,
                jac=_grad_objective,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            if result.success or result.fun < _objective(x0):
                score = math.exp(-result.fun)
                if score > best_score:
                    best_score = score
                    best_params = result.x.copy()
        except Exception:
            continue
    
    # Final polish with very tight tolerance
    if time.time() < deadline - 3.0:
        try:
            result = minimize(
                _objective,
                best_params,
                jac=_grad_objective,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-14}
            )
            if result.fun < _objective(best_params):
                best_params = result.x.copy()
                best_score = math.exp(-result.fun)
        except Exception:
            pass
    
    # Convert to list of [x, y]
    pts = best_params.reshape(8, 2)
    # Ensure coordinates are within bounds
    pts = np.clip(pts, -1e5, 1e5)
    
    return [[float(pts[i, 0]), float(pts[i, 1])] for i in range(8)]
# EVOLVE-BLOCK-END
