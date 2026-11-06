import time
import math
import numpy as np
from scipy.optimize import minimize, basinhopping

_N = 8
_DEADLINE = 150.0
_SAFETY = 15.0
_START = time.time()


def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _START)


def _log_product(pts):
    """Sum of log of all unordered pairwise distances."""
    pts = np.asarray(pts, dtype=float).reshape(_N, 2)
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(d, 1.0)
    return float(np.log(d).sum())


def _diameter(pts):
    pts = np.asarray(pts, dtype=float).reshape(_N, 2)
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    return float(d.max())


def _objective(pts):
    """Minimize negative log-score (scale-invariant, so raw coords are fine)."""
    dia = _diameter(pts)
    if dia < 1e-8:
        return 1e12
    lp = _log_product(pts)
    # log(score) = lp + n(n-1)*log(2) - n(n-1)*log(dia) - n*log(n)
    # constant terms don't matter for minimization; minimize -(lp - n(n-1)log(dia))
    return -(lp - _N * (_N - 1) * math.log(dia))


def _refine(pts):
    """
    Multi-method local refinement: L-BFGS-B -> SLSQP -> Powell -> high-precision L-BFGS-B.
    Each stage is only run if it can improve and time permits.
    """
    pts = np.asarray(pts, dtype=float).reshape(_N, 2)
    best_pts = pts.copy()
    best_obj = _objective(best_pts)

    stages = [
        ('L-BFGS-B', {'maxiter': 400, 'ftol': 1e-10}),
        ('SLSQP', {'maxiter': 150, 'ftol': 1e-12}),
        ('Powell', {'maxiter': 150, 'ftol': 1e-10}),
        ('L-BFGS-B', {'maxiter': 800, 'ftol': 1e-13, 'gtol': 1e-12}),
    ]
    for method, opts in stages:
        if _remaining() < 3.0:
            break
        try:
            res = minimize(_objective, best_pts.ravel(), method=method, options=opts)
            cand = res.x.reshape(_N, 2)
            obj = _objective(cand)
            if obj < best_obj - 1e-12:
                best_pts, best_obj = cand, obj
        except Exception:
            pass
    return best_pts


def _make_cage_init(kind, rng):
    """Generate diverse initial configurations (structural variety)."""
    if kind == 0:  # Concentric squares (outer 45deg, inner 0deg)
        pts = []
        for k in range(4):
            ang = 2 * math.pi * k / 4 + math.pi / 4
            pts.append([1.0 * math.cos(ang), 1.0 * math.sin(ang)])
        for k in range(4):
            ang = 2 * math.pi * k / 4
            pts.append([0.55 * math.cos(ang), 0.55 * math.sin(ang)])
        return np.array(pts, dtype=float)
    if kind == 1:  # Nested rings (outer 5, inner 3)
        pts = []
        for k in range(5):
            ang = 2 * math.pi * k / 5
            pts.append([1.0 * math.cos(ang), 1.0 * math.sin(ang)])
        for k in range(3):
            ang = 2 * math.pi * k / 3
            pts.append([0.5 * math.cos(ang), 0.5 * math.sin(ang)])
        return np.array(pts, dtype=float)
    if kind == 2:  # Deformed octagon
        pts = []
        for k in range(_N):
            ang = 2 * math.pi * k / _N
            r = 1.0 + 0.15 * rng.uniform(-1, 1)
            pts.append([r * math.cos(ang), r * math.sin(ang)])
        return np.array(pts, dtype=float)
    # kind 3: Random
    return rng.uniform(-1.0, 1.0, size=(8, 2))


def _basinhopping_from(x0, seed):
    """Run basin hopping from x0, then refine the result."""
    def bh_obj(x):
        return _objective(x.reshape(_N, 2))
    try:
        bh_res = basinhopping(
            bh_obj, x0.ravel(),
            minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 150, 'ftol': 1e-10}},
            niter=80, seed=seed, stepsize=0.5, T=100.0,
            accept_test=lambda x, fx, x0_, fx0: fx < fx0
        )
        return _refine(bh_res.x.reshape(_N, 2))
    except Exception:
        return None


def construct():
    rng = np.random.default_rng(12345)

    # 1. Regular octagon seed + refinement
    oct_pts = np.array([[math.cos(2 * math.pi * k / _N),
                         math.sin(2 * math.pi * k / _N)] for k in range(_N)], dtype=float)
    best = _refine(oct_pts)
    best_obj = _objective(best)

    # 2. Diverse combinatorial topologies (the structural escape)
    for kind in range(1, 4):
        if _remaining() < 5.0:
            break
        init = _make_cage_init(kind, rng)
        cand = _refine(init)
        obj = _objective(cand)
        if obj < best_obj:
            best, best_obj = cand, obj

    # 3. Basin hopping from current best (primary global escape)
    if _remaining() > 25.0:
        cand = _basinhopping_from(best, seed=12345)
        if cand is not None:
            obj = _objective(cand)
            if obj < best_obj:
                best, best_obj = cand, obj

    # 4. Basin hopping from the strongest non-octagon topology (concentric squares)
    if _remaining() > 25.0:
        init_sq = _make_cage_init(0, rng)
        cand = _basinhopping_from(init_sq, seed=99999)
        if cand is not None:
            obj = _objective(cand)
            if obj < best_obj:
                best, best_obj = cand, obj

    # 5. Extra multi-start random + local refinement while budget remains
    while _remaining() > 12.0:
        init = _make_cage_init(3, rng)
        cand = _refine(init)
        obj = _objective(cand)
        if obj < best_obj:
            best, best_obj = cand, obj

    # 6. Final high-precision refinement of the best configuration
    if _remaining() > 3.0:
        cand = _refine(best)
        obj = _objective(cand)
        if obj < best_obj:
            best, best_obj = cand, obj

    best = np.asarray(best, dtype=float).reshape(_N, 2)

    # Ensure distinct points (nudge any near-duplicates)
    for i in range(_N):
        for j in range(i + 1, _N):
            if np.linalg.norm(best[i] - best[j]) < 1e-5:
                best[j] += [1e-5, 0]

    # Scale to diameter exactly 2 (contract normalization)
    dia = _diameter(best)
    if dia > 1e-12:
        best = best * (2.0 / dia)

    # Bounded coordinates check (should be ~O(1) after scaling)
    return [[float(x), float(y)] for (x, y) in best]
