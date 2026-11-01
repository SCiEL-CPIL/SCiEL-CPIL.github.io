"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 32

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _objective(params, n):
    """
    params: flat array of length 3*n: [x0,y0,r0, x1,y1,r1, ...]
    Objective to MINIMIZE: negative sum of radii + penalty for violations.
    Violations:
      - circles outside [0,1]^2: x < r, x > 1-r, y < r, y > 1-r
      - overlapping circles: dist < r_i + r_j
    We use a smooth penalty (soft-min style via squared penalty).
    """
    pts = params.reshape(n, 3)
    x = pts[:, 0]
    y = pts[:, 1]
    r = pts[:, 2]

    # Ensure radii are positive (small floor)
    r = np.maximum(r, 1e-6)

    sum_r = np.sum(r)

    # Boundary violations
    # For each circle: need x >= r, x <= 1-r, y >= r, y <= 1-r
    # Violation = max(0, r - x) + max(0, r - (1-x)) + max(0, r - y) + max(0, r - (1-y))
    # Use smooth max: log-sum-exp style, but simpler: use (max(0, val))^2
    # Actually, let's use a differentiable penalty.
    
    # Boundary penalty: for each circle, 4 constraints
    # x - r >= 0, (1-x) - r >= 0, y - r >= 0, (1-y) - r >= 0
    # penalty = sum of max(0, -violation)^2
    
    bound_pen = 0.0
    for i in range(n):
        viol_x1 = r[i] - x[i]      # should be <= 0
        viol_x2 = r[i] - (1.0 - x[i])
        viol_y1 = r[i] - y[i]
        viol_y2 = r[i] - (1.0 - y[i])
        # Smooth penalty: use (max(0, viol))^2
        for v in [viol_x1, viol_x2, viol_y1, viol_y2]:
            if v > 0:
                bound_pen += v * v

    # Overlap penalty
    overlap_pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            min_dist = r[i] + r[j]
            if dist_sq < min_dist * min_dist:
                dist = np.sqrt(dist_sq)
                viol = min_dist - dist
                overlap_pen += viol * viol

    total_pen = bound_pen + overlap_pen
    # Scale penalty to be significant
    return -sum_r + 10.0 * total_pen


def _objective_grad(params, n):
    """
    Numerical gradient via finite differences (central).
    """
    eps = 1e-7
    grad = np.zeros_like(params)
    f0 = _objective(params, n)
    for i in range(len(params)):
        params[i] += eps
        f_plus = _objective(params, n)
        params[i] -= 2 * eps
        f_minus = _objective(params, n)
        params[i] += eps
        grad[i] = (f_plus - f_minus) / (2 * eps)
    return grad


def _is_valid(pts, n, tol=1e-9):
    """Check if all circles are inside unit square and non-overlapping."""
    x = pts[:, 0]
    y = pts[:, 1]
    r = pts[:, 2]
    # Boundary check
    if np.any(r < -tol) or np.any(x < r - tol) or np.any(x > 1 - r - tol) \
       or np.any(y < r - tol) or np.any(y > 1 - r - tol):
        return False
    # Overlap check
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True


def _sum_radii(pts):
    return np.sum(pts[:, 2])


def _grid_init(n, jitter=0.0):
    """Create a grid of n circles with optional jitter."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            r = 0.5 / side - 1e-3
            if jitter > 0:
                cx += np.random.uniform(-jitter, jitter)
                cy += np.random.uniform(-jitter, jitter)
                cx = np.clip(cx, r, 1 - r)
                cy = np.clip(cy, r, 1 - r)
            cells.append([cx, cy, r])
    return np.array(cells[:n])


def _optimize_single(pts, n, deadline, method='BFGS'):
    """
    Optimize a single starting configuration.
    Returns (best_pts, best_value, success).
    """
    params = pts.flatten()
    
    # First try BFGS with numerical gradient
    try:
        res = minimize(
            _objective,
            params,
            args=(n,),
            method=method,
            options={'maxiter': 200, 'gtol': 1e-6, 'disp': False}
        )
        if time.time() > deadline - 2:
            return None, None, False
        
        # Check validity of result
        res_pts = res.x.reshape(n, 3)
        # Clip to valid range
        res_pts[:, 0] = np.clip(res_pts[:, 0], 0, 1)
        res_pts[:, 1] = np.clip(res_pts[:, 1], 0, 1)
        res_pts[:, 2] = np.maximum(res_pts[:, 2], 0)
        
        # Enforce boundary constraints
        for i in range(n):
            r = res_pts[i, 2]
            res_pts[i, 0] = np.clip(res_pts[i, 0], r, 1 - r)
            res_pts[i, 1] = np.clip(res_pts[i, 1], r, 1 - r)
            # If radius is too large, shrink it
            max_r = min(res_pts[i, 0], 1 - res_pts[i, 0], res_pts[i, 1], 1 - res_pts[i, 1])
            if res_pts[i, 2] > max_r:
                res_pts[i, 2] = max_r
        
        if _is_valid(res_pts, n):
            return res_pts, _sum_radii(res_pts), True
        else:
            # Try to repair
            for i in range(n):
                r = res_pts[i, 2]
                res_pts[i, 0] = np.clip(res_pts[i, 0], r, 1 - r)
                res_pts[i, 1] = np.clip(res_pts[i, 1], r, 1 - r)
                max_r = min(res_pts[i, 0], 1 - res_pts[i, 0], res_pts[i, 1], 1 - res_pts[i, 1])
                if res_pts[i, 2] > max_r:
                    res_pts[i, 2] = max_r
            if _is_valid(res_pts, n):
                return res_pts, _sum_radii(res_pts), True
    except Exception:
        pass
    
    return None, None, False


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    best_pts = None
    best_value = -np.inf
    
    # Strategy: multiple random restarts with jittered grids
    n_restarts = 15
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5:
            break
        
        # Create initial configuration with jitter
        jitter = 0.05 * (1 - restart / n_restarts)  # Decreasing jitter
        pts = _grid_init(N, jitter=jitter)
        
        # Random perturbation for later restarts
        if restart > 0:
            pts[:, 0] += np.random.uniform(-0.1, 0.1, N)
            pts[:, 1] += np.random.uniform(-0.1, 0.1, N)
            pts[:, 2] *= np.random.uniform(0.8, 1.0, N)
            # Clip to valid range
            for i in range(N):
                r = pts[i, 2]
                pts[i, 0] = np.clip(pts[i, 0], r, 1 - r)
                pts[i, 1] = np.clip(pts[i, 1], r, 1 - r)
        
        # Optimize
        res_pts, res_val, success = _optimize_single(pts, N, deadline)
        
        if success and res_val > best_value:
            best_value = res_val
            best_pts = res_pts.copy()
    
    # Final polish: try to improve the best solution with tighter tolerance
    if best_pts is not None and time.time() < deadline - 3:
        try:
            params = best_pts.flatten()
            res = minimize(
                _objective,
                params,
                args=(N,),
                method='L-BFGS-B',
                options={'maxiter': 100, 'gtol': 1e-8, 'disp': False}
            )
            res_pts = res.x.reshape(N, 3)
            for i in range(N):
                r = res_pts[i, 2]
                res_pts[i, 0] = np.clip(res_pts[i, 0], r, 1 - r)
                res_pts[i, 1] = np.clip(res_pts[i, 1], r, 1 - r)
                max_r = min(res_pts[i, 0], 1 - res_pts[i, 0], res_pts[i, 1], 1 - res_pts[i, 1])
                if res_pts[i, 2] > max_r:
                    res_pts[i, 2] = max_r
            if _is_valid(res_pts, N):
                val = _sum_radii(res_pts)
                if val > best_value:
                    best_value = val
                    best_pts = res_pts
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_pts = np.array(cells[:N])
    
    return best_pts
# EVOLVE-BLOCK-END
