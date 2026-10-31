"""Rewritten program: low-rank sinusoidal deformation of a 4x4 grid optimized via Differential Evolution."""
import time
import numpy as np
from scipy.optimize import differential_evolution

N = 16
D = 2

DEADLINE = 120.0
SAFETY_MARGIN = 5.0  # seconds before hard limit to stop and return best


def _grid_points():
    """Base 4x4 grid centered at origin."""
    coords = np.array(list(np.ndindex(4, 4)), dtype=float)
    # Center it
    coords -= coords.mean(axis=0)
    return coords


def _deform(params, base):
    """
    Apply low-rank sinusoidal deformation to base grid.
    params: [amp_x, freq_x, phase_x, amp_y, freq_y, phase_y, scale]
    """
    amp_x, freq_x, phase_x, amp_y, freq_y, phase_y, scale = params
    x, y = base[:, 0], base[:, 1]
    
    # Orthogonal sinusoidal modes
    dx = amp_x * np.sin(freq_x * y + phase_x)
    dy = amp_y * np.sin(freq_y * x + phase_y)
    
    pts = base * scale
    pts[:, 0] += dx
    pts[:, 1] += dy
    return pts


def _score_points(pts):
    """Compute (max_dist / min_dist)^2. Return inf if degenerate."""
    n = len(pts)
    # Pairwise squared distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    sq_dists = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    min_d = np.min(sq_dists)
    max_d = np.max(sq_dists)
    if min_d <= 1e-12:
        return np.inf
    ratio = max_d / min_d
    return ratio * ratio


def _objective(params):
    """Objective for differential evolution: minimize squared ratio."""
    base = _grid_points()
    pts = _deform(params, base)
    # Check distinctness
    min_d = np.min(_pairwise_sq_dists(pts))
    if min_d < 1e-10:
        return 1e6
    return _score_points(pts)


def _pairwise_sq_dists(pts):
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sum(diff ** 2, axis=2)


def construct():
    start_time = time.time()
    base = _grid_points()
    
    # Initial guess: no deformation, unit scale
    initial_params = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    best_pts = _deform(initial_params, base)
    best_score = _score_points(best_pts)
    
    # Bounds for DE parameters
    # amp: [0, 1], freq: [0.1, 3], phase: [0, 2*pi], scale: [0.5, 2]
    bounds = [
        (0.0, 1.5),   # amp_x
        (0.1, 3.0),   # freq_x
        (0.0, 2 * np.pi),  # phase_x
        (0.0, 1.5),   # amp_y
        (0.1, 3.0),   # freq_y
        (0.0, 2 * np.pi),  # phase_y
        (0.5, 2.0),   # scale
    ]
    
    # Run differential evolution with time budget
    elapsed = time.time() - start_time
    if elapsed < SAFETY_MARGIN:
        try:
            result = differential_evolution(
                _objective,
                bounds,
                maxiter=200,
                popsize=15,
                tol=1e-8,
                seed=42,
                polish=True,
                update_interval=1,
                callback=_de_callback,
                init='sobol',
            )
            if result.success or result.fun < best_score:
                best_params = result.x
                best_pts = _deform(best_params, base)
                best_score = _score_points(best_pts)
        except Exception:
            pass
    
    # Final polish: use Nelder-Mead or L-BFGS on the continuous params
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - SAFETY_MARGIN:
        try:
            from scipy.optimize import minimize
            res = minimize(
                _objective,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10},
            )
            if res.fun < best_score:
                best_params = res.x
                best_pts = _deform(best_params, base)
                best_score = _score_points(best_pts)
        except Exception:
            pass
    
    # Ensure distinctness
    min_d = np.min(_pairwise_sq_dists(best_pts))
    if min_d < 1e-10:
        # Fallback to grid
        best_pts = base.astype(float)
    
    return best_pts


def _de_callback(xk, convergence=False):
    """Callback to check time budget during DE."""
    if time.time() > 120.0 - SAFETY_MARGIN:
        raise TimeoutError("Time budget exceeded")
