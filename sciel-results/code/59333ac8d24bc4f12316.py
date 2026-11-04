"""Heilbronn convex regions: maximize min triangle area / hull area."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

# Precompute all triangle index triplets
_TRIPLETS = np.array(list(combinations(range(N), 3)))
_N_TRIPLES = _TRIPLETS.shape[0]
_I, _J, _K = _TRIPLETS.T

def _triangle_areas(pts):
    """Compute all C(n,3) triangle areas vectorized."""
    pi = pts[_I]
    pj = pts[_J]
    pk = pts[_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    return 0.5 * np.abs(cross)

def _hull_area(pts):
    try:
        return ConvexHull(pts).volume
    except Exception:
        return 0.0

def _score(pts):
    h = _hull_area(pts)
    if h <= 1e-15:
        return 0.0
    return float(np.min(_triangle_areas(pts)) / h)

def _neg_soft_min(x, beta):
    """Smooth surrogate: minimize -(softmin of triangle areas)/hull_area."""
    pts = x.reshape(N, 2)
    h = _hull_area(pts)
    if h <= 1e-15:
        return 1e6
    areas = _triangle_areas(pts)
    scaled = -areas * beta
    m = np.max(scaled)
    soft_min = -m / beta - np.log(np.sum(np.exp(scaled - m))) / beta
    return -soft_min / h

def _regular_polygon(n, radius=1.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _two_ring_params(n_outer, n_inner, r_inner):
    """Initial parameterization: outer regular polygon + inner ring."""
    ao = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    ai = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + np.pi / n_inner
    outer = np.column_stack([np.cos(ao), np.sin(ao)])
    inner = np.column_stack([r_inner * np.cos(ai), r_inner * np.sin(ai)])
    return np.vstack([outer, inner])

def _halton(n, base):
    seq = np.zeros(n)
    f = 1.0
    for i in range(1, n + 1):
        f /= base
        s = 0.0
        j = i
        while j > 0:
            s += (j % base) * f
            j //= base
        seq[i - 1] = s
    return seq

def _halton_points(n):
    x = _halton(n, 2)
    y = _halton(n, 3)
    return np.column_stack([x, y])

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    best_score = 0.0
    best_pts = _regular_polygon(N, 1.0).copy()

    def update_best(pts, score):
        nonlocal best_score, best_pts
        if score > best_score:
            best_score = score
            best_pts = np.array(pts, dtype=float).copy()

    rng = np.random.default_rng(123)

    # Build a diverse pool of initial candidates
    pool = []
    # Regular polygon
    pool.append(_regular_polygon(N, 1.0))
    # Two-ring configurations with various splits and radii
    for n_outer in [6, 7, 8, 9, 10, 11]:
        n_inner = N - n_outer
        if n_inner >= 3:
            for r_inner in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                pool.append(_two_ring_params(n_outer, n_inner, r_inner))
    # Halton low-discrepancy
    h = _halton_points(N)
    pool.append(h * 2.0)
    pool.append(h * 2.0 + 0.5)
    # Perturbed regular polygon
    for i in range(12):
        pool.append(_regular_polygon(N, 1.0) + rng.normal(0, 0.05, (N, 2)))
    # Random in disk
    for i in range(10):
        a = rng.uniform(0, 2 * np.pi, N)
        r = rng.uniform(0.3, 1.0, N)
        pool.append(np.column_stack([r * np.cos(a), r * np.sin(a)]))

    # Score the pool, keep the best few
    scored = []
    for pts in pool:
        if time.time() > deadline:
            break
        s = _score(pts)
        update_best(pts, s)
        scored.append((s, pts))
    scored.sort(key=lambda x: -x[0])
    top = [pts for _, pts in scored[:8]]

    # Global exploration: differential evolution on a smooth surrogate
    if time.time() < deadline - 40:
        bounds = [(-1.5, 1.5)] * (2 * N)
        try:
            de_res = differential_evolution(
                _neg_soft_min,
                bounds,
                args=(40.0,),
                seed=7,
                maxiter=40,
                tol=1e-6,
                popsize=12,
                polish=False,
                init='sobol',
                update_interval=1,
            )
            de_pts = de_res.x.reshape(N, 2)
            s = _score(de_pts)
            update_best(de_pts, s)
            top = [de_pts.copy()] + top[:6]
        except Exception:
            pass

    # Local refinement: L-BFGS with progressive beta (smooth -> sharp)
    for beta in [30.0, 80.0, 200.0, 500.0]:
        if time.time() > deadline - 12:
            break
        for init_pts in top[:6]:
            if time.time() > deadline - 12:
                break
            try:
                res = minimize(
                    _neg_soft_min,
                    init_pts.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 400, 'ftol': 1e-13, 'gtol': 1e-9, 'maxls': 60},
                )
                res_pts = res.x.reshape(N, 2)
                s = _score(res_pts)
                update_best(res_pts, s)
            except Exception:
                pass
        # Refresh top with current best
        top = [best_pts.copy()] + top[:5]

    # Final exact-score polish via Nelder-Mead on the best
    if time.time() < deadline - 8:
        try:
            res = minimize(
                lambda x: -_score(x.reshape(N, 2)),
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'xatol': 1e-7, 'fatol': 1e-9, 'maxiter': 3000},
            )
            res_pts = res.x.reshape(N, 2)
            s = _score(res_pts)
            update_best(res_pts, s)
        except Exception:
            pass

    # One more high-beta L-BFGS pass if time remains
    if time.time() < deadline - 4:
        try:
            res = minimize(
                _neg_soft_min,
                best_pts.flatten(),
                args=(800.0,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50},
            )
            res_pts = res.x.reshape(N, 2)
            s = _score(res_pts)
            update_best(res_pts, s)
        except Exception:
            pass

    return best_pts
# EVOLVE-BLOCK-END
