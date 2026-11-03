"""Circle packing: 50 equal circles in unit square [-1/2,1/2]^2.
Design: SLSQP linear-epigraph (maximize t s.t. exact pairwise & boundary
constraints) from a dense hexagonal start, with restarts and a polish pass.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
HALF = 0.5

# ---- precompute pair indices ----
_PAIRS = [(i, j) for i in range(N) for j in range(i + 1, N)]

# ---- constraint set (built once) ----
def _build_constraints():
    cons = []
    for (i, j) in _PAIRS:
        ii, jj = 2 * i, 2 * j
        def fun(x, ii=ii, jj=jj):
            return np.hypot(x[ii] - x[jj], x[ii + 1] - x[jj + 1]) - 2.0 * x[-1]
        def jac(x, ii=ii, jj=jj):
            g = np.zeros_like(x)
            dx = x[ii] - x[jj]; dy = x[ii + 1] - x[jj + 1]
            nrm = np.hypot(dx, dy)
            if nrm < 1e-12:
                nrm = 1e-12
            g[ii] = dx / nrm; g[ii + 1] = dy / nrm
            g[jj] = -dx / nrm; g[jj + 1] = -dy / nrm
            g[-1] = -2.0
            return g
        cons.append({'type': 'ineq', 'fun': fun, 'jac': jac})
    for i in range(N):
        for k in range(2):
            idx = 2 * i + k
            def fun(x, idx=idx):
                return (HALF - abs(x[idx])) - x[-1]
            def jac(x, idx=idx):
                g = np.zeros_like(x)
                g[idx] = -1.0 if x[idx] >= 0 else 1.0
                g[-1] = -1.0
                return g
            cons.append({'type': 'ineq', 'fun': fun, 'jac': jac})
    return cons

_CONSTS = _build_constraints()
_BOUNDS = [(-HALF, HALF)] * (2 * N) + [(0.0, HALF)]

def _obj(x):
    return -x[-1]

def _obj_grad(x):
    g = np.zeros_like(x)
    g[-1] = -1.0
    return g

# ---- true (exact) min-radius of a configuration ----
def _radius(coords):
    r = (HALF - np.abs(coords)).min()
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, 1)
    r = min(r, dist[iu].min() / 2.0)
    return float(r)

# ---- initial guesses ----
def _grid_initial():
    m = 8
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _hex_initial():
    dx = 0.14
    dy = dx * np.sqrt(3.0) / 2.0
    pts = []
    row = 0
    y = -HALF + dx / 2.0
    while len(pts) < N and row < 200:
        off = (dx / 2.0) if (row % 2) else 0.0
        x = -HALF + dx / 2.0 + off
        while x <= HALF and len(pts) < N:
            pts.append([x, y])
            x += dx
        y += dy
        row += 1
    return np.array(pts[:N])

def construct():
    t0 = time.time()
    BUDGET = 110.0          # safety margin well under the 120s / 150s limits
    best = None
    best_r = -1.0

    def run(x0, maxiter, ftol):
        return minimize(_obj, x0, method='SLSQP', jac=_obj_grad,
                        bounds=_BOUNDS, constraints=_CONSTS,
                        options={'maxiter': maxiter, 'ftol': ftol})

    def consider(coords):
        nonlocal best, best_r
        coords = np.clip(coords, -HALF, HALF)
        r = _radius(coords)
        if r > best_r:
            best_r = r
            best = coords.copy()

    # candidate starts: dense hex, square grid, and small random perturbations
    rng = np.random.default_rng(0)
    starts = [
        _hex_initial(),
        _grid_initial(),
        _hex_initial() + 0.012 * rng.standard_normal((N, 2)),
        _grid_initial() + 0.012 * rng.standard_normal((N, 2)),
    ]

    for init in starts:
        if time.time() - t0 > BUDGET - 25.0:
            break
        x0 = np.concatenate([init.ravel(), [0.05]])
        try:
            res = run(x0, 120, 1e-12)
            consider(res.x[:2 * N].reshape(N, 2))
        except Exception:
            continue

    # high-precision polish on the best configuration found
    if best is not None and time.time() - t0 < BUDGET - 10.0:
        x0 = np.concatenate([best.ravel(), [max(best_r, 1e-4)]])
        try:
            res = run(x0, 350, 1e-15)
            consider(res.x[:2 * N].reshape(N, 2))
        except Exception:
            pass

    if best is None:
        best = _grid_initial()
    return best
