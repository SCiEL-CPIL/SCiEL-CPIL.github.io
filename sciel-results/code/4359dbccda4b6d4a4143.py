import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi, Delaunay
import cma

N = 26
EPS = 1e-8
DEADLINE = 145.0


def _penalty_objective(params, penalty=5000.0):
    """Penalty objective for CMA-ES: minimize -sum(r) + penalty*(overlaps + wall violations)."""
    p = params.reshape(N, 3)
    x = p[:, 0]
    y = p[:, 1]
    r = p[:, 2]

    score = -np.sum(r)

    wx = np.maximum(0.0, r - x) + np.maximum(0.0, r - (1.0 - x))
    wy = np.maximum(0.0, r - y) + np.maximum(0.0, r - (1.0 - y))
    wall = np.sum(wx) + np.sum(wy)

    iu = np.triu_indices(N, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx * dx + dy * dy)
    overlap = np.maximum(0.0, (r[iu[0]] + r[iu[1]]) - dist)
    pair = np.sum(overlap)

    return score + penalty * (pair + wall)


def _enforce_validity(params, max_iter=200):
    """Enforce all constraints: walls and pairwise non-overlap by shrinking radii."""
    p = params.reshape(N, 3).copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)

    wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                      np.minimum(p[:, 1], 1.0 - p[:, 1]))
    p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))

    for _ in range(max_iter):
        iu = np.triu_indices(N, k=1)
        dx = p[iu[0], 0] - p[iu[1], 0]
        dy = p[iu[0], 1] - p[iu[1], 1]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = (p[iu[0], 2] + p[iu[1], 2]) - dist
        active = overlap > 0
        if not np.any(active):
            break
        for k in range(len(active)):
            if not active[k]:
                continue
            i = iu[0][k]
            j = iu[1][k]
            ov = overlap[k]
            if p[i, 2] <= p[j, 2]:
                shrink = min(ov, p[i, 2] - 1e-6)
                p[i, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[j, 2] - 1e-6)
                    p[j, 2] -= shrink2
            else:
                shrink = min(ov, p[j, 2] - 1e-6)
                p[j, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[i, 2] - 1e-6)
                    p[i, 2] -= shrink2
        wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                          np.minimum(p[:, 1], 1.0 - p[:, 1]))
        p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))

    return p


def _score_valid(p):
    return np.sum(p[:, 2])


def _contact_graph_refine(p, time_budget=8.0, seed=42):
    """
    Refine a valid configuration using Delaunay contact-graph-aware local optimization.
    Focus optimization on local interactions (contact graph) for tighter packings.
    """
    start = time.time()
    best_p = p.copy()
    best_score = _score_valid(p)
    rng = np.random.RandomState(seed)

    def _local_penalty(params, penalty=2000.0):
        pp = params.reshape(N, 3)
        x, y, r = pp[:, 0], pp[:, 1], pp[:, 2]
        score = -np.sum(r)
        wx = np.maximum(0.0, r - x) + np.maximum(0.0, r - (1.0 - x))
        wy = np.maximum(0.0, r - y) + np.maximum(0.0, r - (1.0 - y))
        wall = np.sum(wx) + np.sum(wy)
        iu = np.triu_indices(N, k=1)
        dx = x[iu[0]] - x[iu[1]]
        dy = y[iu[0]] - y[iu[1]]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = np.maximum(0.0, (r[iu[0]] + r[iu[1]]) - dist)
        pair = np.sum(overlap)
        return score + penalty * (pair + wall)

    def _local_grad(params, penalty=2000.0):
        pp = params.reshape(N, 3)
        x, y, r = pp[:, 0], pp[:, 1], pp[:, 2]
        n = N
        g = np.zeros_like(params)
        g[2::3] = -1.0
        for i in range(n):
            if r[i] - x[i] > 0:
                g[3*i] += penalty * (-1.0)
                g[3*i+2] += penalty * (+1.0)
            if r[i] - (1.0 - x[i]) > 0:
                g[3*i] += penalty * (+1.0)
                g[3*i+2] += penalty * (+1.0)
            if r[i] - y[i] > 0:
                g[3*i+1] += penalty * (-1.0)
                g[3*i+2] += penalty * (+1.0)
            if r[i] - (1.0 - y[i]) > 0:
                g[3*i+1] += penalty * (+1.0)
                g[3*i+2] += penalty * (+1.0)
        iu = np.triu_indices(n, k=1)
        dx = x[iu[0]] - x[iu[1]]
        dy = y[iu[0]] - y[iu[1]]
        dist = np.sqrt(dx * dx + dy * dy)
        dist_safe = np.maximum(dist, 1e-10)
        ov = (r[iu[0]] + r[iu[1]]) - dist
        active = ov > 0
        if np.any(active):
            ax = iu[0][active]
            ay = iu[1][active]
            adx = dx[active] / dist_safe[active]
            ady = dy[active] / dist_safe[active]
            for k in range(len(ax)):
                i = ax[k]
                j = ay[k]
                g[3*i] += penalty * (-adx[k])
                g[3*j] += penalty * (+adx[k])
                g[3*i+1] += penalty * (-ady[k])
                g[3*j+1] += penalty * (+ady[k])
                g[3*i+2] += penalty * (+1.0)
                g[3*j+2] += penalty * (+1.0)
        return g

    lo = np.zeros(3 * N)
    hi = np.zeros(3 * N)
    for i in range(N):
        lo[3*i] = 0.0
        hi[3*i] = 1.0
        lo[3*i+1] = 0.0
        hi[3*i+1] = 1.0
        lo[3*i+2] = 1e-6
        hi[3*i+2] = 0.5
    bounds_list = list(zip(lo, hi))

    # Iteratively: optimize, enforce validity, grow radii, repeat
    current = best_p.copy()
    for it in range(60):
        if time.time() - start > time_budget:
            break
        try:
            res = minimize(
                _local_penalty,
                current.flatten(),
                method='L-BFGS-B',
                jac=_local_grad,
                bounds=bounds_list,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            p_new = _enforce_validity(res.x)
            s_new = _score_valid(p_new)
            if s_new > best_score:
                best_score = s_new
                best_p = p_new.copy()
                current = p_new.copy()
                # Try to grow radii slightly
                current[:, 2] *= 1.005
                current = _enforce_validity(current)
            else:
                # Nudge with small random perturbation
                current = best_p.copy()
                current[:, :2] += rng.normal(0, 0.005, size=(N, 2))
                current = _enforce_validity(current)
        except Exception:
            break

    return best_p


def _cma_es_search(x0, sigma0, time_budget=30.0, seed=42):
    """Run CMA-ES to explore the configuration space globally."""
    start = time.time()
    x0 = np.array(x0, dtype=float)
    n = len(x0)

    # Bounds for CMA-ES
    lo = np.zeros(n)
    hi = np.zeros(n)
    for i in range(N):
        lo[3*i] = 0.0
        hi[3*i] = 1.0
        lo[3*i+1] = 0.0
        hi[3*i+1] = 1.0
        lo[3*i+2] = 1e-6
        hi[3*i+2] = 0.5

    # Population size
    popsize = max(4, int(4 + 3 * np.log(n)))

    options = {
        'bounds': [lo, hi],
        'seed': seed,
        'maxiter': 200,
        'verbose': -9,
        'tolopt': 1e-9,
        'tolx': 1e-9,
        'popsize': popsize,
    }

    try:
        es = cma.CMAEvolutionStrategy(x0, sigma0, options)
        while not es.stop() and (time.time() - start) < time_budget:
            solutions = es.ask()
            fits = [_penalty_objective(s) for s in solutions]
            es.tell(solutions, fits)
        result = es.result.xbest
    except Exception:
        result = x0

    return result


def _grid_initial(seed):
    rng = np.random.RandomState(seed)
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4))
    cells = cells[:N]
    arr = np.array(cells, dtype=float).reshape(-1)
    arr += rng.uniform(-1e-5, 1e-5, size=arr.size)
    return arr


def _hex_initial(seed):
    rng = np.random.RandomState(seed)
    points = []
    for i in range(7):
        for j in range(7):
            x = i
            y = j * 1.5 + (0.5 if i % 2 else 0)
            points.append((x, y))
    points = np.array(points)
    points -= np.mean(points, axis=0)
    max_coord = np.max(np.abs(points))
    scale = 0.45 / max_coord
    points *= scale
    points += 0.5
    dists = np.sum((points - 0.5)**2, axis=1)
    indices = np.argsort(dists)[:N]
    p = points[indices]
    radii = np.full(N, 0.05)
    params = np.empty(3 * N)
    params[0::3] = p[:, 0]
    params[1::3] = p[:, 1]
    params[2::3] = radii
    params += rng.uniform(-1e-4, 1e-4, size=params.size)
    return params


def _voronoi_initial(seed):
    rng = np.random.RandomState(seed)
    points = rng.uniform(0.05, 0.95, size=(N, 2))
    radii = np.zeros(N)
    for i, p in enumerate(points):
        dists = np.linalg.norm(points - p, axis=1)
        dists[i] = np.inf
        min_dist_neighbor = np.min(dists)
        wall_dist = np.min([p[0], 1-p[0], p[1], 1-p[1]])
        radii[i] = 0.5 * np.min([min_dist_neighbor, wall_dist]) * 0.95
    params = np.empty(3 * N)
    params[0::3] = points[:, 0]
    params[1::3] = points[:, 1]
    params[2::3] = radii
    params += rng.uniform(-1e-5, 1e-5, size=params.size)
    return params


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    best_params = None
    best_score = -np.inf

    # Generate diverse starting points
    initializers = [
        lambda s: _grid_initial(s),
        lambda s: _hex_initial(s),
        lambda s: _voronoi_initial(s),
    ]

    # Phase 1: CMA-ES global exploration from multiple starts
    seeds = [42, 123, 7, 999, 555, 2024]
    cma_candidates = []

    for seed in seeds:
        if time.time() > deadline - 60:
            break
        for init_fn in initializers:
            if time.time() > deadline - 60:
                break
            x0 = init_fn(seed)
            x0 = _enforce_validity(x0).flatten()
            # CMA-ES search
            cma_result = _cma_es_search(x0, sigma0=0.05, time_budget=10.0, seed=seed)
            p_cma = _enforce_validity(cma_result)
            s_cma = _score_valid(p_cma)
            cma_candidates.append((s_cma, p_cma))
            if s_cma > best_score:
                best_score = s_cma
                best_params = p_cma.copy()

    # Phase 2: Contact-graph refinement of top candidates
    cma_candidates.sort(key=lambda x: -x[0])
    top_candidates = cma_candidates[:8]

    for i, (s_cma, p_cma) in enumerate(top_candidates):
        if time.time() > deadline - 10:
            break
        # Refine with contact-graph-aware local optimization
        p_refined = _contact_graph_refine(p_cma, time_budget=8.0, seed=42 + i)
        s_refined = _score_valid(p_refined)
        if s_refined > best_score:
            best_score = s_refined
            best_params = p_refined.copy()

    # Phase 3: Shake-and-refine to escape local minima
    if best_params is not None and time.time() < deadline - 5:
        rng = np.random.RandomState(12345)
        sigma = 0.015
        for _ in range(3000):
            if time.time() > deadline - 3:
                break
            p_new = best_params.copy()
            p_new[:, :2] += rng.normal(0, sigma, size=(N, 2))
            p_new[:, 2] += rng.normal(0, sigma * 0.3, size=N)
            p_new = _enforce_validity(p_new)
            s_new = _score_valid(p_new)
            if s_new > best_score:
                best_score = s_new
                best_params = p_new.copy()
                # Occasionally refine the new best
                if time.time() < deadline - 8:
                    p_ref = _contact_graph_refine(p_new, time_budget=4.0, seed=int(rng.randint(1e6)))
                    s_ref = _score_valid(p_ref)
                    if s_ref > best_score:
                        best_score = s_ref
                        best_params = p_ref.copy()

    # Fallback
    if best_params is None:
        best_params = _enforce_validity(_grid_initial(42))

    # Final: try to grow radii
    for _ in range(20):
        if time.time() > deadline - 1:
            break
        p = best_params.copy()
        p[:, 2] *= 1.002
        p_new = _enforce_validity(p)
        s_new = _score_valid(p_new)
        if s_new > best_score:
            best_score = s_new
            best_params = p_new
        else:
            break

    return best_params.copy()
