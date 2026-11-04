"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26
DEADLINE_S = 115.0  # safety margin below 120s (evaluator kills at 150s)


def _objective_and_constraints(params, n):
    """
    params: array of shape (3*n,) -> [x0, y0, r0, x1, y1, r1, ...]
    Returns (objective, constraints_list)
    Objective: -sum(r_i)  (we minimize negative sum of radii)
    Constraints:
      - x_i - r_i >= 0
      - y_i - r_i >= 0
      - 1 - x_i - r_i >= 0
      - 1 - y_i - r_i >= 0
      - r_i > 0 (enforced via soft lower bound in objective penalty)
      - dist(i,j) - r_i - r_j >= 0 for all i < j
    """
    pts = params.reshape(n, 3)
    xs, ys, rs = pts[:, 0], pts[:, 1], pts[:, 2]
    
    obj = -np.sum(rs)
    
    # Penalty for negative radii or out-of-bounds to help SLSQP
    # (hard constraints below handle the real bounds)
    penalty = 0.0
    penalty += np.sum(np.maximum(0, -rs)) * 1000
    penalty += np.sum(np.maximum(0, -xs + rs)) * 1000
    penalty += np.sum(np.maximum(0, -ys + rs)) * 1000
    penalty += np.sum(np.maximum(0, xs + rs - 1)) * 1000
    penalty += np.sum(np.maximum(0, ys + rs - 1)) * 1000
    
    obj += penalty
    
    constraints = []
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[3*i] - p[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[3*i+1] - p[3*i+2]})
    
    for i in range(n):
        for j in range(i+1, n):
            xi, yi, ri = p = None  # placeholder
            # Use closure over i,j
            ii, jj = i, j
            def _dist_con(p, ii=ii, jj=jj):
                di = np.sqrt((p[3*ii]-p[3*jj])**2 + (p[3*ii+1]-p[3*jj+1])**2)
                return di - p[3*ii+2] - p[3*jj+2]
            constraints.append({'type': 'ineq', 'fun': _dist_con})
    
    return obj, constraints


def _objective_only(params, n):
    """Fast objective for SLSQP (constraints handled separately)."""
    pts = params.reshape(n, 3)
    xs, ys, rs = pts[:, 0], pts[:, 1], pts[:, 2]
    return -np.sum(rs)


def _make_constraint_funs(n):
    """Return list of constraint functions (inequality: fun(p) >= 0)."""
    cons = []
    for i in range(n):
        cons.append(lambda p, i=i: p[3*i] - p[3*i+2])
        cons.append(lambda p, i=i: p[3*i+1] - p[3*i+2])
        cons.append(lambda p, i=i: 1 - p[3*i] - p[3*i+2])
        cons.append(lambda p, i=i: 1 - p[3*i+1] - p[3*i+2])
    for i in range(n):
        for j in range(i+1, n):
            def _dc(p, i=i, j=j):
                return np.sqrt((p[3*i]-p[3*j])**2 + (p[3*i+1]-p[3*j+1])**2) - p[3*i+2] - p[3*j+2]
            cons.append(_dc)
    return cons


def _is_valid(arr, tol=1e-9):
    """Check if a (n,3) array represents valid disjoint circles in [0,1]^2."""
    if arr.shape != (N, 3):
        return False
    xs, ys, rs = arr[:, 0], arr[:, 1], arr[:, 2]
    if np.any(rs <= 0):
        return False
    if np.any(xs < -tol) or np.any(ys < -tol) or np.any(xs > 1+tol) or np.any(ys > 1+tol):
        return False
    if np.any(xs - rs < -tol) or np.any(ys - rs < -tol) or np.any(xs + rs - 1 > tol) or np.any(ys + rs - 1 > tol):
        return False
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
            if d < rs[i] + rs[j] - tol:
                return False
    return True


def _sum_radii(arr):
    return float(np.sum(arr[:, 2]))


def _grid_init(n, jitter=0.0):
    """Grid-based initialization."""
    side = int(np.ceil(np.sqrt(n)))
    arr = np.zeros((n, 3))
    for k in range(n):
        i = k % side
        j = k // side
        arr[k, 0] = (i + 0.5) / side
        arr[k, 1] = (j + 0.5) / side
        arr[k, 2] = 0.5 / side - 0.001
    if jitter > 0:
        arr[:, 0] += np.random.uniform(-jitter, jitter, n)
        arr[:, 1] += np.random.uniform(-jitter, jitter, n)
    return arr


def _random_init(n, rng):
    """Random initialization with small circles."""
    arr = np.zeros((n, 3))
    arr[:, 0] = rng.uniform(0.1, 0.9, n)
    arr[:, 1] = rng.uniform(0.1, 0.9, n)
    arr[:, 2] = rng.uniform(0.02, 0.06, n)
    return arr


def _optimize_from(arr, maxiter=300, time_limit=20.0):
    """Run SLSQP from initial guess arr. Returns best valid result or None."""
    n = arr.shape[0]
    x0 = arr.flatten().copy()
    cons_funs = _make_constraint_funs(n)
    
    bounds = []
    for i in range(n):
        bounds.append((0, 1))  # x
        bounds.append((0, 1))  # y
        bounds.append((1e-6, 0.5))  # r
    
    constraints = [{'type': 'ineq', 'fun': f} for f in cons_funs]
    
    start = time.time()
    best_val = -_sum_radii(arr)
    best_x = x0.copy()
    
    try:
        res = minimize(
            _objective_only, x0, args=(n,),
            method='SLSQP', bounds=bounds,
            constraints=constraints,
            options={'maxiter': maxiter, 'ftol': 1e-12}
        )
        elapsed = time.time() - start
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
    except Exception:
        pass
    
    # Convert back and validate
    best_arr = best_x.reshape(n, 3)
    # Clamp to valid range
    best_arr[:, 0] = np.clip(best_arr[:, 0], 0, 1)
    best_arr[:, 1] = np.clip(best_arr[:, 1], 0, 1)
    best_arr[:, 2] = np.clip(best_arr[:, 2], 1e-6, 0.5)
    
    if _is_valid(best_arr):
        return best_arr
    return None


def _polish(arr, maxiter=200, time_limit=10.0):
    """High-precision polish pass with tighter tolerance."""
    n = arr.shape[0]
    x0 = arr.flatten().copy()
    cons_funs = _make_constraint_funs(n)
    
    bounds = []
    for i in range(n):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((1e-8, 0.5))
    
    constraints = [{'type': 'ineq', 'fun': f} for f in cons_funs]
    
    try:
        res = minimize(
            _objective_only, x0, args=(n,),
            method='SLSQP', bounds=bounds,
            constraints=constraints,
            options={'maxiter': maxiter, 'ftol': 1e-14}
        )
        best_x = res.x
    except Exception:
        best_x = x0
    
    best_arr = best_x.reshape(n, 3)
    best_arr[:, 0] = np.clip(best_arr[:, 0], 0, 1)
    best_arr[:, 1] = np.clip(best_arr[:, 1], 0, 1)
    best_arr[:, 2] = np.clip(best_arr[:, 2], 1e-8, 0.5)
    
    if _is_valid(best_arr):
        return best_arr
    return None


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_arr = _grid_init(N, jitter=0.0)
    best_val = _sum_radii(best_arr)
    
    # Phase 1: Multi-start with diverse initializations
    n_starts = 25
    for s in range(n_starts):
        if time.time() - start_time > DEADLINE_S - 30:
            break
        
        if s == 0:
            arr = _grid_init(N, jitter=0.0)
        elif s == 1:
            arr = _grid_init(N, jitter=0.02)
        elif s == 2:
            arr = _grid_init(N, jitter=0.05)
        else:
            arr = _random_init(N, rng)
        
        result = _optimize_from(arr, maxiter=250, time_limit=15.0)
        if result is not None:
            val = _sum_radii(result)
            if val > best_val:
                best_val = val
                best_arr = result.copy()
    
    # Phase 2: Polish the best result
    if time.time() - start_time < DEADLINE_S - 15:
        polished = _polish(best_arr, maxiter=300, time_limit=15.0)
        if polished is not None:
            val = _sum_radii(polished)
            if val > best_val:
                best_val = val
                best_arr = polished.copy()
    
    # Final validation
    if not _is_valid(best_arr):
        # Fallback to grid
        best_arr = _grid_init(N, jitter=0.0)
    
    return best_arr
