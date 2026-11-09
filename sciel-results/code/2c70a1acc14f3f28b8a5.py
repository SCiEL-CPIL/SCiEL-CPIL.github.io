# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

TOL = 1e-7

def _check_valid(centers, sides, tol=1e-7):
    n = len(sides)
    if centers.shape != (n, 2) or sides.shape != (n,):
        return False
    if not (np.all(np.isfinite(centers)) and np.all(np.isfinite(sides))):
        return False
    if np.any(sides <= 0):
        return False
    left = centers[:, 0] - sides / 2
    right = centers[:, 0] + sides / 2
    bottom = centers[:, 1] - sides / 2
    top = centers[:, 1] + sides / 2
    if np.any(left < -tol) or np.any(right > 1 + tol):
        return False
    if np.any(bottom < -tol) or np.any(top > 1 + tol):
        return False
    cx = centers[:, 0]; cy = centers[:, 1]
    dx = np.abs(cx[:, None] - cx[None, :])
    dy = np.abs(cy[:, None] - cy[None, :])
    t = (sides[:, None] + sides[None, :]) / 2 - tol
    iu = np.triu_indices(n, k=1)
    if np.any((dx[iu] < t[iu]) & (dy[iu] < t[iu])):
        return False
    return True

def _softplus(z, k=8.0):
    z = np.clip(z, -30, 30)
    return np.where(z > 1.0 / k, z, np.log(1.0 + np.exp(k * z)) / k)

def _objective_neg(x, n=10, k=8.0):
    s = x[2::3]
    return -np.sum(s) + k * np.sum(_softplus(x[3 * n:]))

def _make_constraints(n=10, tol=1e-7, k=8.0):
    M = 2.0
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    idx = {(i, j): 3 * n + (len(pairs) * 4 + m) for m, (i, j) in enumerate(pairs)}
    constraints = []
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 2 * x[3 * i] - x[3 * i + 2] - tol})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 2 * (1 - x[3 * i]) - x[3 * i + 2] - tol})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 2 * x[3 * i + 1] - x[3 * i + 2] - tol})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 2 * (1 - x[3 * i + 1]) - x[3 * i + 2] - tol})
    for (i, j) in pairs:
        b = idx[(i, j)]
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i, j=j, b=b: (x[3 * j] - x[3 * i]) - 0.5 * (x[3 * i + 2] + x[3 * j + 2]) + tol + M * (1 - x[b])})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i, j=j, b=b: (x[3 * i] - x[3 * j]) - 0.5 * (x[3 * i + 2] + x[3 * j + 2]) + tol + M * (1 - x[b + 1])})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i, j=j, b=b: (x[3 * j + 1] - x[3 * i + 1]) - 0.5 * (x[3 * i + 2] + x[3 * j + 2]) + tol + M * (1 - x[b + 2])})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i, j=j, b=b: (x[3 * i + 1] - x[3 * j + 1]) - 0.5 * (x[3 * i + 2] + x[3 * j + 2]) + tol + M * (1 - x[b + 3])})
    return constraints

def _slsqp_binary(x0, n=10, tol=1e-7, k=8.0, maxiter=300, time_limit=15, constraints=None):
    if constraints is None:
        constraints = _make_constraints(n, tol, k)
    nb = 4 * (n * (n - 1) // 2)
    bounds = [(0, 1), (0, 1), (1e-6, 1)] * n + [(0, 1)] * nb
    st = time.time()
    try:
        r = minimize(_objective_neg, x0, args=(n, k), method='SLSQP', bounds=bounds,
                     constraints=constraints, options={'maxiter': maxiter, 'ftol': 1e-10})
        if r.x is not None and np.all(np.isfinite(r.x)):
            return r.x, time.time() - st
    except Exception:
        pass
    return None, time.time() - st

def _extract(x, n):
    cx = np.clip(x[0::3], 0, 1)
    cy = np.clip(x[1::3], 0, 1)
    s = np.clip(x[2::3], 1e-6, 1.0)
    for i in range(n):
        cx[i] = np.clip(cx[i], s[i] / 2, 1 - s[i] / 2)
        cy[i] = np.clip(cy[i], s[i] / 2, 1 - s[i] / 2)
    return np.column_stack([cx, cy]), s

def _local_improve(centers, sides, tol=1e-7, max_iters=5000, step_size=0.01, time_limit=30, seed=42):
    n = len(sides)
    best_centers = centers.copy(); best_sides = sides.copy(); best_val = np.sum(sides)
    if not _check_valid(best_centers, best_sides, tol):
        return best_centers, best_sides, best_val
    rng = np.random.default_rng(seed)
    st = time.time(); T = 0.01; cooling = 0.999; min_T = 1e-6
    for _ in range(max_iters):
        if time.time() - st > time_limit or T < min_T:
            break
        idx = rng.integers(0, n)
        nc = best_centers.copy(); ns = best_sides.copy()
        noise = step_size * (0.5 + 0.5 * T)
        nc[idx, 0] += rng.uniform(-noise, noise)
        nc[idx, 1] += rng.uniform(-noise, noise)
        ns[idx] = max(1e-6, min(1.0, ns[idx] + rng.uniform(-noise, noise)))
        s = ns[idx]
        nc[idx, 0] = np.clip(nc[idx, 0], s / 2, 1 - s / 2)
        nc[idx, 1] = np.clip(nc[idx, 1], s / 2, 1 - s / 2)
        if _check_valid(nc, ns, tol):
            nv = np.sum(ns); d = nv - best_val
            if d > 0 or rng.random() < np.exp(d / T):
                best_centers, best_sides, best_val = nc, ns, nv
        T *= cooling
    return best_centers, best_sides, best_val

def _init_2_large_8_small():
    n = 10; c = np.zeros((n, 2)); s = np.zeros(n)
    c[0] = [0.25, 0.25]; s[0] = 0.5
    c[1] = [0.75, 0.25]; s[1] = 0.5
    for k in range(8):
        col, row = k % 4, k // 4
        c[2 + k] = [(col + 0.5) * 0.25, 0.5 + (row + 0.5) * 0.25]; s[2 + k] = 0.25
    return c, s

def _init_grid():
    n = 10; c = np.zeros((n, 2)); s = np.full(n, 0.25)
    for i in range(n):
        col, row = i % 4, i // 4
        c[i] = [(col + 0.5) * 0.25, (row + 0.5) * 0.25]
    return c, s

def _init_mixed():
    n = 10; c = np.zeros((n, 2)); s = np.zeros(n)
    c[0] = [0.25, 0.25]; s[0] = 0.5
    for k in range(6):
        col, row = k % 2, k // 2
        c[1 + k] = [0.5 + (col + 0.5) * 0.25, (row + 0.5) * (1 / 3)]; s[1 + k] = 1 / 3
    for k in range(3):
        c[7 + k] = [(k + 0.5) * (0.5 / 3), 0.5 + 0.5 / 6]; s[7 + k] = 1 / 3
    return c, s

def _to_x(centers, sides, n=10, bin_mode='x'):
    x = np.zeros(3 * n + 4 * (n * (n - 1) // 2))
    x[0::3] = centers[:, 0]; x[1::3] = centers[:, 1]; x[2::3] = sides
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    for m, (i, j) in enumerate(pairs):
        b = 3 * n + (len(pairs) * 4 + m)
        if bin_mode == 'x':
            x[b] = 1.0
        elif bin_mode == 'y':
            x[b + 2] = 1.0
        else:
            # choose the direction with most slack
            dx = centers[j, 0] - centers[i, 0]
            dy = centers[j, 1] - centers[i, 1]
            if dx >= 0 and dx >= dy:
                x[b] = 1.0
            elif dx < 0 and -dx >= abs(dy):
                x[b + 1] = 1.0
            elif dy >= 0:
                x[b + 2] = 1.0
            else:
                x[b + 3] = 1.0
    return x

def squares_in_square10() -> tuple:
    n = 10; tol = 1e-7
    start = time.time(); deadline = 600
    best_centers = None; best_sides = None; best_val = -1.0

    def _try(centers, sides):
        nonlocal best_centers, best_sides, best_val
        if not _check_valid(centers, sides, tol):
            return
        v = np.sum(sides)
        if v > best_val:
            best_val, best_centers, best_sides = v, centers.copy(), sides.copy()

    inits = [_init_2_large_8_small, _init_grid, _init_mixed]
    for f in inits:
        if time.time() - start > deadline - 60:
            break
        _try(*f())
    for f in inits:
        if time.time() - start > deadline - 60:
            break
        c, s = f()
        if _check_valid(c, s, tol):
            _try(*_local_improve(c, s, tol, max_iters=50000, time_limit=45, seed=1))

    # MILP-style binary optimization (exact non-overlap via big-M)
    rng = np.random.default_rng(123)
    cons = _make_constraints(n, tol)
    base = _init_2_large_8_small()
    for attempt in range(25):
        if time.time() - start > deadline - 60:
            break
        if attempt == 0:
            c0, s0 = base
        elif attempt < 4:
            c0, s0 = inits[attempt - 1]()
        elif best_centers is not None:
            c0 = best_centers + rng.uniform(-0.04, 0.04, (n, 2))
            s0 = np.clip(best_sides + rng.uniform(-0.04, 0.04, n), 0.01, 1.0)
            for i in range(n):
                c0[i, 0] = np.clip(c0[i, 0], s0[i] / 2, 1 - s0[i] / 2)
                c0[i, 1] = np.clip(c0[i, 1], s0[i] / 2, 1 - s0[i] / 2)
        else:
            c0 = rng.uniform(0.1, 0.9, (n, 2)); s0 = rng.uniform(0.15, 0.4, n)
            for i in range(n):
                c0[i, 0] = np.clip(c0[i, 0], s0[i] / 2, 1 - s0[i] / 2)
                c0[i, 1] = np.clip(c0[i, 1], s0[i] / 2, 1 - s0[i] / 2)
        mode = ['x', 'y', 'auto', 'x', 'y'][attempt % 5]
        x0 = _to_x(c0, s0, n, mode)
        xo, _ = _slsqp_binary(x0, n, tol, k=8.0, maxiter=300, time_limit=12, constraints=cons)
        if xo is not None:
            _try(*_extract(xo, n))

    # continuous softplus relaxation (fallback / different basin)
    for attempt in range(8):
        if time.time() - start > deadline - 60:
            break
        if best_centers is not None and attempt > 0:
            c0 = best_centers + rng.uniform(-0.05, 0.05, (n, 2))
            s0 = np.clip(best_sides + rng.uniform(-0.05, 0.05, n), 0.01, 1.0)
            for i in range(n):
                c0[i, 0] = np.clip(c0[i, 0], s0[i] / 2, 1 - s0[i] / 2)
                c0[i, 1] = np.clip(c0[i, 1], s0[i] / 2, 1 - s0[i] / 2)
        else:
            c0, s0 = base
        x0 = _to_x(c0, s0, n, 'auto')
        xo, _ = _slsqp_binary(x0, n, tol, k=3.0, maxiter=200, time_limit=12, constraints=cons)
        if xo is not None:
            _try(*_extract(xo, n))

    # random-restart local polish
    if best_centers is not None:
        for r in range(8):
            if time.time() - start > deadline - 30:
                break
            c_p = best_centers + rng.uniform(-0.05, 0.05, (n, 2))
            s_p = np.clip(best_sides + rng.uniform(-0.05, 0.05, n), 0.01, 1.0)
            for i in range(n):
                c_p[i, 0] = np.clip(c_p[i, 0], s_p[i] / 2, 1 - s_p[i] / 2)
                c_p[i, 1] = np.clip(c_p[i, 1], s_p[i] / 2, 1 - s_p[i] / 2)
            if _check_valid(c_p, s_p, tol):
                _try(*_local_improve(c_p, s_p, tol, max_iters=20000, time_limit=25, seed=100 + r))

    # final high-precision polish
    if best_centers is not None and time.time() - start < deadline - 10:
        _try(*_local_improve(best_centers, best_sides, tol, max_iters=50000, step_size=0.005, time_limit=60, seed=7))

    if best_centers is None:
        best_centers, best_sides = _init_2_large_8_small()
    return best_centers, best_sides

# EVOLVE-BLOCK-END
