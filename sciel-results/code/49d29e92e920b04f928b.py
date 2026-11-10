# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 630.0
_TIME_MARGIN = 15.0  # safety margin before evaluator kill

# Precompute grid points
_XS = np.arange(-3.0, 1.5, 0.01)
_YS = np.arange(-1.5, 1.5, 0.01)
_X, _Y = np.meshgrid(_XS, _YS, indexing='ij')
_GX = _X.ravel()
_GY = _Y.ravel()


def _densify_path(path):
    """Densify a path the same way the evaluator does.
    Prepends (0,0,0) and appends (-2,0,90).
    Between consecutive rows, adds linear interpolations in m = max(1, ceil(|a2-a1|)) steps.
    """
    full = np.vstack([[0.0, 0.0, 0.0], path, [-2.0, 0.0, 90.0]])
    result = [full[0]]
    for i in range(len(full) - 1):
        p1 = full[i]
        p2 = full[i + 1]
        da = abs(p2[2] - p1[2])
        m = max(1, int(np.ceil(da)))
        for k in range(1, m + 1):
            t = k / m
            interp = p1 + t * (p2 - p1)
            result.append(interp)
    return np.array(result)


def _check_in_base(u, v):
    """Check if points (u, v) are inside the base L-shaped corridor.
    Base = {0 <= u <= 1, -3 <= v <= 1} U {-3 <= u <= 1, 0 <= v <= 1}
    """
    arm1 = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
    arm2 = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
    return arm1 | arm2


def _evaluate_path(path):
    """Evaluate the sofa area for a given path.
    Returns the number of grid points inside every placement (times 0.0001).
    """
    full = _densify_path(path)
    n_pts = len(_GX)
    inside = np.ones(n_pts, dtype=bool)
    for row in full:
        x, y, a = row[0], row[1], row[2]
        rad = -np.deg2rad(a)
        c = np.cos(rad)
        s = np.sin(rad)
        # w - (x,y)
        wx = _GX - x
        wy = _GY - y
        # R(-a) * (w - (x,y))
        u = c * wx - s * wy
        v = s * wx + c * wy
        inside &= _check_in_base(u, v)
        if not inside.any():
            break
    return inside.sum() * 0.0001


def _nelder_mead(func, x0, bounds=None, max_iter=500, tol=1e-6, deadline=None):
    """Simple Nelder-Mead optimizer with deadline support."""
    n = len(x0)
    simplex = [x0.astype(np.float64)]
    for i in range(n):
        x = x0.copy()
        x[i] = x0[i] + 0.1
        simplex.append(x)
    simplex = np.array(simplex)
    fvals = np.array([func(v) for v in simplex])
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - _TIME_MARGIN:
            break
        order = np.argsort(fvals)
        simplex = simplex[order]
        fvals = fvals[order]
        
        centroid = simplex[:-1].mean(axis=0)
        alpha = 1.0
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = func(xr)
        
        if fr < fvals[0]:
            gamma = 2.0
            xe = centroid + gamma * (centroid - simplex[-1])
            fe = func(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-1]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            beta = 0.5
            xc = centroid + beta * (simplex[-1] - centroid)
            fc = func(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = func(simplex[i])
        
        if np.max(fvals) - np.min(fvals) < tol:
            break
    
    best_idx = np.argmin(fvals)
    return simplex[best_idx], fvals[best_idx]


def _local_search(path, n_iter=50, step_size=0.05, deadline=None):
    """Random local search around a path."""
    best_path = path.copy()
    best_val = _evaluate_path(best_path)
    
    for i in range(n_iter):
        if deadline is not None and time.time() > deadline - _TIME_MARGIN:
            break
        trial = best_path + np.random.randn(*best_path.shape) * step_size
        # Keep angles monotonic-ish and in range
        trial[:, 2] = np.clip(trial[:, 2], 0.0, 90.0)
        # Ensure angles are roughly increasing
        trial[:, 2] = np.maximum.accumulate(trial[:, 2])
        val = _evaluate_path(trial)
        if val > best_val:
            best_val = val
            best_path = trial
    
    return best_path, best_val


def moving_sofa_2d() -> np.ndarray:
    """
    The 2D moving-sofa problem: find the largest-area rigid shape that can be moved
    through an L-shaped corridor of width 1 that makes a single 90-degree turn.

    Returns:
        path: np.ndarray of shape (n, 3); each row is (x, y, angle_degrees) for the
            corridor's inner corner.
    """
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # Initial hand-tuned path
    path = np.array(
        [
            [-0.1, 0.1, 15.0],
            [-0.2, 0.2, 20.0],
            [-0.3, 0.3, 34.0],
            [-0.3, 0.3, 40.0],
            [-0.4, 0.3, 50.0],
            [-0.5, 0.2, 55.0],
            [-0.6, 0.2, 75.0],
            [-0.7, 0.3, 86.0],
        ],
        dtype=np.float64,
    )
    
    best_path = path.copy()
    best_val = _evaluate_path(best_path)
    
    # Phase 1: Nelder-Mead on the path parameters
    x0 = best_path.ravel()
    x_opt, f_opt = _nelder_mead(
        lambda x: _evaluate_path(x.reshape(-1, 3)),
        x0,
        max_iter=200,
        deadline=deadline,
    )
    if f_opt > best_val:
        best_val = f_opt
        best_path = x_opt.reshape(-1, 3)
    
    # Phase 2: Local random search with decreasing step size
    for step in [0.1, 0.05, 0.02, 0.01]:
        if time.time() > deadline - _TIME_MARGIN:
            break
        best_path, best_val = _local_search(
            best_path, n_iter=30, step_size=step, deadline=deadline
        )
    
    # Ensure valid output
    best_path = np.asarray(best_path, dtype=np.float64)
    if best_path.shape[0] < 1:
        best_path = path
    if not np.all(np.isfinite(best_path)):
        best_path = path
    
    return best_path


# EVOLVE-BLOCK-END
