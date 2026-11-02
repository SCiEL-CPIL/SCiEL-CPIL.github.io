"""Circle packing in unit square: 26 disjoint circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26
EPS = 1e-6


def _objective(params):
    """Penalty objective: minimize  -sum(r) + penalty*overlaps.

    params layout: [x0,y0,r0, x1,y1,r1, ...]
    """
    p = params.reshape(N, 3)
    x = p[:, 0]
    y = p[:, 1]
    r = p[:, 2]

    # keep radii positive (small lower bound)
    r_eff = np.maximum(r, 1e-4)

    score = -np.sum(r_eff)

    # wall boundary violations
    wx = np.maximum(0.0, r_eff - x) + np.maximum(0.0, r_eff - (1.0 - x))
    wy = np.maximum(0.0, r_eff - y) + np.maximum(0.0, r_eff - (1.0 - y))
    wall = np.sum(wx) + np.sum(wy)

    # pairwise overlap violations
    n = N
    iu = np.triu_indices(n, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx * dx + dy * dy)
    overlap = np.maximum(0.0, (r_eff[iu[0]] + r_eff[iu[1]]) - dist)
    pair = np.sum(overlap)

    penalty = 200.0
    return score + penalty * (pair + wall)


def _objective_grad(params):
    """Gradient of _objective (numerical fallback not needed; analytic below)."""
    p = params.reshape(N, 3)
    x = p[:, 0]
    y = p[:, 1]
    r = p[:, 2]
    r_eff = np.maximum(r, 1e-4)
    n = N

    g = np.zeros_like(params)

    # d(-sum r)/d r = -1
    g[2::3] = -1.0

    penalty = 200.0

    # wall contributions
    # x lower: max(0, r - x) -> if r - x > 0: d/dx = -1, d/dr = +1
    # x upper: max(0, r - (1 - x)) -> if r - (1-x) > 0: d/dx = +1, d/dr = +1
    for i in range(n):
        if r_eff[i] - x[i] > 0:
            g[3 * i + 0] += penalty * (-1.0)
            g[3 * i + 2] += penalty * (+1.0)
        if r_eff[i] - (1.0 - x[i]) > 0:
            g[3 * i + 0] += penalty * (+1.0)
            g[3 * i + 2] += penalty * (+1.0)
        if r_eff[i] - y[i] > 0:
            g[3 * i + 1] += penalty * (-1.0)
            g[3 * i + 2] += penalty * (+1.0)
        if r_eff[i] - (1.0 - y[i]) > 0:
            g[3 * i + 1] += penalty * (+1.0)
            g[3 * i + 2] += penalty * (+1.0)

    # pairwise
    iu = np.triu_indices(n, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx * dx + dy * dy)
    dist_safe = np.maximum(dist, 1e-8)
    ov = (r_eff[iu[0]] + r_eff[iu[1]]) - dist
    active = ov > 0
    if np.any(active):
        ax = iu[0][active]
        ay = iu[1][active]
        adx = dx[active] / dist_safe[active]
        ady = dy[active] / dist_safe[active]
        # overlap = (ri+rj) - dist ; d/dxi = -adx, d/dxj = +adx ; d/dri = +1, d/drj = +1
        for k in range(len(ax)):
            i = ax[k]; j = ay[k]
            g[3 * i + 0] += penalty * (-adx[k])
            g[3 * j + 0] += penalty * (+adx[k])
            g[3 * i + 1] += penalty * (-ady[k])
            g[3 * j + 1] += penalty * (+ady[k])
            g[3 * i + 2] += penalty * (+1.0)
            g[3 * j + 2] += penalty * (+1.0)

    return g


def _grid_initial(seed):
    """Grid of equal circles, first N used."""
    rng = np.random.RandomState(seed)
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3))
    cells = cells[:N]
    arr = np.array(cells, dtype=float).reshape(-1)
    # tiny jitter
    arr += rng.uniform(-1e-4, 1e-4, size=arr.size)
    return arr


def _random_initial(seed):
    """Random feasible-ish positions with modest radii."""
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.1, 0.9, size=N)
    y = rng.uniform(0.1, 0.9, size=N)
    r = np.full(N, 0.1)
    params = np.empty(3 * N)
    params[0::3] = x
    params[1::3] = y
    params[2::3] = r
    return params


def _bounds():
    lo = np.zeros(3 * N)
    hi = np.zeros(3 * N)
    for i in range(N):
        lo[3 * i + 0] = 0.0
        hi[3 * i + 0] = 1.0
        lo[3 * i + 1] = 0.0
        hi[3 * i + 1] = 1.0
        lo[3 * i + 2] = 1e-4
        hi[3 * i + 2] = 0.5
    return (lo, hi)


def _refine(params, deadline, rng, budget_per_opt):
    """Run L-BFGS-B with decreasing penalties, then stochastic refinement."""
    lo, hi = _bounds()
    best = params.copy()
    best_val = _objective(params)

    # multi-penalty L-BFGS-B passes
    for penalty in [200.0, 500.0, 1200.0]:
        if time.time() > deadline:
            break
        # temporarily set module-level penalty by patching objective via closure
        # We re-implement objective with given penalty here to keep it clean.
        def obj(p, pen=penalty):
            q = p.reshape(N, 3)
            x = q[:, 0]; y = q[:, 1]; r = q[:, 2]
            r_eff = np.maximum(r, 1e-4)
            score = -np.sum(r_eff)
            wx = np.maximum(0.0, r_eff - x) + np.maximum(0.0, r_eff - (1.0 - x))
            wy = np.maximum(0.0, r_eff - y) + np.maximum(0.0, r_eff - (1.1 - y) * 0 - (1.0 - y))
            wall = np.sum(wx) + np.sum(wy)
            iu = np.triu_indices(N, k=1)
            dx = x[iu[0]] - x[iu[1]]; dy = y[iu[0]] - y[iu[1]]
            dist = np.sqrt(dx * dx + dy * dy)
            ov = np.maximum(0.0, (r_eff[iu[0]] + r_eff[iu[1]]) - dist)
            return score + pen * (np.sum(ov) + wall)

        res = minimize(obj, params, method='L-BFGS-B', bounds=list(zip(lo, hi)),
                       options={'maxiter': budget_per_opt, 'maxfun': budget_per_opt})
        params = res.x
        val = _objective(params)
        if val < best_val:
            best_val = val
            best = params.copy()

    # stochastic / simulated-annealing refinement on the best
    params = best.copy()
    val = best_val
    T = 0.05
    step = 0.02
    iters = 4000
    for it in range(iters):
        if time.time() > deadline:
            break
        # pick a random circle
        i = rng.randint(N)
        # random perturbation of x,y,r
        cand = params.copy()
        cand[3 * i + 0] = params[3 * i + 0] + rng.uniform(-step, step)
        cand[3 * i + 1] = params[3 * i + 1] + rng.uniform(-step, step)
        cand[3 * i + 2] = params[3 * i + 2] + rng.uniform(-step * 0.5, step * 0.5)
        # clamp
        cand[3 * i + 0] = np.clip(cand[3 * i + 0], 0.0, 1.0)
        cand[3 * i + 1] = np.clip(cand[3 * i + 1], 0.0, 1.0)
        cand[3 * i + 2] = np.clip(cand[3 * i + 2], 1e-4, 0.5)
        new_val = _objective(cand)
        d = new_val - val
        if d < 0 or rng.rand() < np.exp(-d / max(T, 1e-6)):
            params = cand
            val = new_val
            if val < best_val:
                best_val = val
                best = params.copy()
        T *= 0.999
        step *= 0.9995

    # final local polish
    if time.time() < deadline:
        res = minimize(_objective, best, jac=_objective_grad, method='L-BFGS-B',
                       bounds=list(zip(lo, hi)),
                       options={'maxiter': 200, 'maxfun': 200})
        if _objective(res.x) < best_val:
            best = res.x
            best_val = _objective(best)

    return best, best_val


def _score_and_clamp(params):
    """Return a valid (x,y,r) array: enforce boundaries and non-overlap by shrinking."""
    p = params.reshape(N, 3).copy()
    # clamp centers
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    # clamp radii to wall distance
    wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                      np.minimum(p[:, 1], 1.0 - p[:, 1]))
    p[:, 2] = np.clip(p[:, 2], 1e-6, np.maximum(wall, 1e-6))

    # iteratively resolve overlaps by shrinking the smaller circle
    for _ in range(200):
        iu = np.triu_indices(N, k=1)
        dx = p[iu[0], 0] - p[iu[1], 0]
        dy = p[iu[0], 1] - p[iu[1], 1]
        dist = np.sqrt(dx * dx + dy * dy)
        need = p[iu[0], 2] + p[iu[1], 2] - dist
        active = need > EPS
        if not np.any(active):
            break
        for k in range(len(active)):
            if not active[k]:
                continue
            i = iu[0][k]
            j = iu[1][k]
            gap = need[k]
            # shrink proportionally to radius, but keep both positive
            tot = p[i, 2] + p[j, 2]
            if tot <= EPS:
                continue
            # reduce each by half the gap, limited by its radius
            shrink_i = min(p[i, 2] - 1e-6, gap * 0.5)
            shrink_j = min(p[j, 2] - 1e-6, gap - shrink_i)
            if shrink_i < 0:
                shrink_i = 0.0
            if shrink_j < 0:
                shrink_j = 0.0
            p[i, 2] -= shrink_i
            p[j, 2] -= shrink_j
            # re-clamp to walls
            p[i, 2] = min(p[i, 2], np.maximum(np.minimum(np.minimum(p[i, 0], 1 - p[i, 0]),
                                                         np.minimum(p[i, 1], 1 - p[i, 1])), 1e-6))
            p[j, 2] = min(p[j, 2], np.maximum(np.minimum(np.minimum(p[j, 0], 1 - p[j, 0]),
                                                         np.minimum(p[j, 1], 1 - p[j, 1])), 1e-6))
    return p


def construct():
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin before 150s kill

    rng = np.random.RandomState(12345)

    # candidate initializations
    inits = []
    inits.append(_grid_initial(0))
    for s in range(1, 6):
        inits.append(_random_initial(s))

    best_params = None
    best_val = np.inf
    best_score = 0.0

    for idx, init in enumerate(inits):
        if time.time() > deadline:
            break
        # allocate remaining time across remaining inits
        remaining = deadline - time.time()
        n_left = len(inits) - idx
        per_init = max(1.0, remaining / max(n_left, 1))
        budget_per_opt = int(min(2000, max(300, per_init * 1000)))
        params, val = _refine(init, deadline, rng, budget_per_opt)
        # compute actual feasible score
        feas = _score_and_clamp(params)
        sc = float(np.sum(feas[:, 2]))
        if sc > best_score:
            best_score = sc
            best_params = feas

    # If we somehow have nothing, fall back to grid
    if best_params is None:
        grid = _grid_initial(0)
        best_params = _score_and_clamp(grid)
        best_score = float(np.sum(best_params[:, 2]))

    return best_params


N = 26
