"""Circle packing in unit square: pack 26 disjoint circles in [0,1]^2 maximizing sum of radii."""
import numpy as np
import time

N = 26
DEADLINE_MARGIN = 5.0  # seconds to stay under the 120s (or 150s) kill limit
_START = time.time()


def _time_left():
    return DEADLINE_MARGIN - (time.time() - _START)


def _softplus(z):
    """Numerically stable softplus = log(1 + exp(z))."""
    return np.where(z > 30.0, z, np.log1p(np.exp(np.clip(z, -30.0, 30.0))))


def _pairwise_indices(n):
    iu = np.triu_indices(n, k=1)
    return iu[0], iu[1]


def _build_problem(n):
    idx_i, idx_j = _pairwise_indices(n)
    return idx_i, idx_j


def _objective_and_grad(v, idx_i, idx_j, soft_beta, boundary_beta):
    """
    v: (3n,) flattened (x,y,r) per circle.
    Returns (loss, grad) for MINIMIZATION: loss = -sum(r) + beta_soft * softplus(overlaps) + beta_b * softplus(boundary).
    softplus(g) with g = violation amount (positive means infeasible).
    """
    n = len(v) // 3
    P = v.reshape(n, 3)
    x = P[:, 0]; y = P[:, 1]; r = P[:, 2]

    # ---- Pairwise overlap: g_ij = (r_i + r_j) - dist_ij ; want g_ij <= 0
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist = np.sqrt(dx * dx + dy * dy + 1e-12)
    g_pair = (r[idx_i] + r[idx_j]) - dist  # positive => overlap

    # softplus penalty
    sp = _softplus(soft_beta * g_pair)
    # d softplus(soft_beta * g)/dg = soft_beta * sigmoid(soft_beta * g)
    sg = np.exp(soft_beta * g_pair) / (1.0 + np.exp(soft_beta * g_pair))
    dL_dg = soft_beta * sg  # d loss / d g

    # grad w.r.t. positions/radii
    # g = r_i + r_j - dist
    # d g / d x_i = -dx/dist, d g / d y_i = -dy/dist
    # d g / d x_j = +dx/dist, d g / d y_j = +dy/dist
    # d g / d r_i = +1, d g / d r_j = +1
    inv_dist = 1.0 / dist
    gx_i = -dx * inv_dist
    gy_i = -dy * inv_dist
    gx_j = dx * inv_dist
    gy_j = dy * inv_dist

    grad = np.zeros_like(v)
    # accumulate
    # contribution = dL_dg * d g / d var
    contrib = dL_dg

    # circle i
    grad[idx_i * 3 + 0] += contrib * gx_i
    grad[idx_i * 3 + 1] += contrib * gy_i
    grad[idx_i * 3 + 2] += contrib * 1.0
    # circle j
    grad[idx_j * 3 + 0] += contrib * gx_j
    grad[idx_j * 3 + 1] += contrib * gy_j
    grad[idx_j * 3 + 2] += contrib * 1.0

    # ---- Boundary: g = r - x (x>=r), g = r - (1-x) => (r + x -1) (x<=1-r)
    g_b = np.concatenate([
        r - x,          # x >= r
        r + x - 1.0,    # x <= 1 - r
        r - y,
        r + y - 1.0,
    ])
    sp_b = _softplus(boundary_beta * g_b)
    sg_b = np.exp(boundary_beta * g_b) / (1.0 + np.exp(boundary_beta * g_b))
    dL_dg_b = boundary_beta * sg_b

    # accumulate boundary grads
    # g = r - x  -> d/dx=-1, d/dr=+1
    grad[:n * 3 + 0] += dL_dg_b[0:n] * (-1.0)
    grad[:n * 3 + 2] += dL_dg_b[0:n] * 1.0
    # g = r + x - 1 -> d/dx=+1, d/dr=+1
    grad[:n * 3 + 0] += dL_dg_b[n:2 * n] * 1.0
    grad[:n * 3 + 2] += dL_dg_b[n:2 * n] * 1.0
    # g = r - y
    grad[:n * 3 + 1] += dL_dg_b[2 * n:3 * n] * (-1.0)
    grad[:n * 3 + 2] += dL_dg_b[2 * n:3 * n] * 1.0
    # g = r + y - 1
    grad[:n * 3 + 1] += dL_dg_b[3 * n:4 * n] * 1.0
    grad[:n * 3 + 2] += dL_dg_b[3 * n:4 * n] * 1.0

    # objective term: -sum(r) -> grad w.r.t. r = -1
    grad[2::3] += -1.0

    loss = -np.sum(r) + np.sum(sp) + np.sum(sp_b)
    return loss, grad


def _validity(v):
    """Check disjoint interiors and inside square. Returns (valid, sum_r)."""
    n = len(v) // 3
    P = v.reshape(n, 3)
    x = P[:, 0]; y = P[:, 1]; r = P[:, 2]
    if np.any(r < 0):
        return False, 0.0
    # boundary with tiny tolerance
    eps = 1e-7
    if np.any(x - r < -eps) or np.any(x + r - 1.0 > eps) or np.any(y - r < -eps) or np.any(y + r - 1.0 > eps):
        return False, 0.0
    # pairwise
    iu = np.triu_indices(n, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx * dx + dy * dy)
    if np.any(dist < (r[iu[0]] + r[iu[1]]) - eps):
        return False, 0.0
    return True, float(np.sum(r))


def _grid_seed(n, rng):
    side = int(np.ceil(np.sqrt(n)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
             for i in range(side) for j in range(side)]
    arr = np.array(cells[:n])
    # small jitter to break symmetry
    arr[:, :2] += rng.uniform(-1e-3, 1e-3, size=(n, 2))
    return arr


def _random_seed(n, rng):
    # random non-overlapping-ish start: place in grid then jitter
    side = int(np.ceil(np.sqrt(n)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
             for i in range(side) for j in range(side)]
    arr = np.array(cells[:n])
    arr[:, :2] += rng.uniform(-0.05, 0.05, size=(n, 2))
    arr[:, 2] *= rng.uniform(0.8, 1.0, size=n)
    return arr


def _optimize_one(v0, idx_i, idx_j, max_iters, deadline_fn, polish=False):
    from scipy.optimize import minimize
    n = len(v0) // 3
    if polish:
        soft_beta = 60.0
        boundary_beta = 60.0
        gtol = 1e-9
        ftol = 1e-12
        iters = min(max_iters, 500)
    else:
        soft_beta = 20.0
        boundary_beta = 20.0
        gtol = 1e-6
        ftol = 1e-9
        iters = max_iters

    def fun(v):
        if deadline_fn() < 0:
            # signal stop by returning a large value; L-BFGS will still finish step
            pass
        loss, g = _objective_and_grad(v, idx_i, idx_j, soft_beta, boundary_beta)
        return loss, g

    # bounds: x,y in [0,1], r in [0, 0.5]
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 0.5))

    res = minimize(fun, v0, jac=True, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': iters, 'gtol': gtol, 'ftol': ftol,
                            'maxfun': iters * 4, 'disp': False})
    return res.x


def construct():
    global _START
    _START = time.time()
    rng = np.random.default_rng(20240501)
    idx_i, idx_j = _build_problem(N)

    best_v = None
    best_sum = -1.0

    # Start from grid (valid) as a safe baseline
    grid = _grid_seed(N, rng)
    valid, s = _validity(grid)
    if valid:
        best_v = grid.copy()
        best_sum = s

    def deadline():
        return _time_left()

    # Phase 1: optimize the grid start
    if deadline() > 3.0:
        v = _optimize_one(grid.ravel(), idx_i, idx_j, max_iters=200, deadline_fn=deadline, polish=False)
        valid, s = _validity(v)
        if valid and s > best_sum:
            best_v = v.reshape(N, 3)
            best_sum = s

    # Phase 2: multiple random restarts with polish
    restarts = 14
    for k in range(restarts):
        if deadline() < 6.0:
            break
        seed = _random_seed(N, rng)
        valid, s = _validity(seed)
        if not valid:
            # project: shrink radii a bit to make valid
            seed[:, 2] *= 0.9
            valid, s = _validity(seed)
        if not valid:
            continue
        v = _optimize_one(seed.ravel(), idx_i, idx_j, max_iters=120, deadline_fn=deadline, polish=False)
        valid, s = _validity(v)
        if valid and s > best_sum:
            best_v = v.reshape(N, 3)
            best_sum = s

    # Phase 3: high-precision polish on the best
    if best_v is not None and deadline() > 4.0:
        v = _optimize_one(best_v.ravel(), idx_i, idx_j, max_iters=500, deadline_fn=deadline, polish=True)
        valid, s = _validity(v)
        if valid and s > best_sum:
            best_v = v.reshape(N, 3)
            best_sum = s

    if best_v is None:
        # absolute fallback: the trivial grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best_v = np.array(cells[:N])

    # Final safety: ensure validity; if not, fall back to grid
    valid, s = _validity(best_v)
    if not valid:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best_v = np.array(cells[:N])

    return best_v
