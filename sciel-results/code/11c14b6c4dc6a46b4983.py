import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

_N = 8
_DEADLINE = 150.0
_SAFETY = 15.0
_START = time.time()


def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _START)


def _diameter(pts):
    pts = np.asarray(pts, dtype=float).reshape(_N, 2)
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    return float(d.max())


def _log_product(pts):
    pts = np.asarray(pts, dtype=float).reshape(_N, 2)
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(d, 1.0)
    return float(np.log(d).sum())


def _score_from_log_product_and_diameter(lp, dia):
    n = _N
    return math.exp(lp + n * (n - 1) * (math.log(2.0) - math.log(dia)) - n * math.log(n))


def _score(pts):
    dia = _diameter(pts)
    if dia <= 1e-12:
        return 0.0
    return _score_from_log_product_and_diameter(_log_product(pts), dia)


def _objective(pts):
    dia = _diameter(pts)
    if dia < 1e-8:
        return 1e12
    return -_score_from_log_product_and_diameter(_log_product(pts), dia)


def _refine(pts):
    """
    Multi-stage local refinement.
    1. L-BFGS-B with standard tolerances.
    2. SLSQP for different landscape navigation.
    3. Final high-precision L-BFGS-B.
    """
    pts = np.asarray(pts, dtype=float).reshape(_N, 2)
    best_pts = pts.copy()
    best_score = _score(pts)

    # Stage 1: Standard L-BFGS-B
    try:
        res = minimize(_objective, pts.ravel(), method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10})
        cand = res.x.reshape(_N, 2)
        s = _score(cand)
        if s > best_score:
            best_pts, best_score = cand, s
    except Exception:
        pass

    # Stage 2: SLSQP
    if _remaining() > 5.0:
        try:
            res = minimize(_objective, best_pts.ravel(), method='SLSQP', options={'maxiter': 200, 'ftol': 1e-12})
            cand = res.x.reshape(_N, 2)
            s = _score(cand)
            if s > best_score:
                best_pts, best_score = cand, s
        except Exception:
            pass

    # Stage 3: High-precision polish
    if _remaining() > 2.0:
        try:
            res = minimize(_objective, best_pts.ravel(), method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
            cand = res.x.reshape(_N, 2)
            s = _score(cand)
            if s > best_score:
                best_pts, best_score = cand, s
        except Exception:
            pass

    return best_pts, best_score


def _make_cage_init(kind, rng):
    """Generate diverse initial configurations."""
    if kind == 0:  # Concentric squares
        a = 1.0
        b = 0.5
        pts = []
        for k in range(4):
            ang = 2 * math.pi * k / 4 + math.pi / 4
            pts.append([a * math.cos(ang), a * math.sin(ang)])
        for k in range(4):
            ang = 2 * math.pi * k / 4
            pts.append([b * math.cos(ang), b * math.sin(ang)])
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
    if kind == 3:  # Two circles
        pts = []
        for k in range(4):
            ang = 2 * math.pi * k / 4
            pts.append([1.0 * math.cos(ang), 1.0 * math.sin(ang)])
        for k in range(4):
            ang = 2 * math.pi * k / 4 + math.pi / 4
            pts.append([0.6 * math.cos(ang), 0.6 * math.sin(ang)])
        return np.array(pts, dtype=float)
    # kind 4: Random
    return rng.uniform(-1.0, 1.0, size=(8, 2))


def construct():
    t0 = time.time()
    rng = np.random.default_rng(42)

    # 1. Seed with regular octagon and refine
    oct_pts = np.array([[math.cos(2 * math.pi * k / _N),
                         math.sin(2 * math.pi * k / _N)] for k in range(_N)], dtype=float)
    best, best_score = _refine(oct_pts)

    # 2. Refine diverse combinatorial topologies
    for kind in range(1, 5):
        if _remaining() < 5.0:
            break
        init = _make_cage_init(kind, rng)
        cand, s = _refine(init)
        if s > best_score:
            best, best_score = cand, s

    # 3. Differential Evolution for global search
    if _remaining() > 30.0:
        def de_objective(x):
            return _objective(x.reshape(_N, 2))

        bounds = [(-2.0, 2.0)] * (2 * _N)
        try:
            de_res = differential_evolution(
                de_objective, bounds, seed=42, maxiter=100, popsize=20,
                tol=1e-8, polish=False, init='sobol'
            )
            cand = de_res.x.reshape(_N, 2)
            cand, s = _refine(cand)
            if s > best_score:
                best, best_score = cand, s
        except Exception:
            pass

    # 4. More DE runs with different seeds if time permits
    if _remaining() > 30.0:
        for seed in [123, 456]:
            if _remaining() < 30.0:
                break
            try:
                de_res = differential_evolution(
                    de_objective, bounds, seed=seed, maxiter=80, popsize=15,
                    tol=1e-8, polish=False, init='latinhypercube'
                )
                cand = de_res.x.reshape(_N, 2)
                cand, s = _refine(cand)
                if s > best_score:
                    best, best_score = cand, s
            except Exception:
                pass

    # 5. Extra multi-start random + local refinement while budget remains
    while _remaining() > 10.0:
        init = _make_cage_init(4, rng)
        cand, s = _refine(init)
        if s > best_score:
            best, best_score = cand, s

    # Final high-precision refinement of the best configuration
    if _remaining() > 2.0:
        cand, s = _refine(best)
        if s > best_score:
            best, best_score = cand, s

    best = np.asarray(best, dtype=float).reshape(_N, 2)

    # Ensure distinct points and bounded coordinates
    for i in range(_N):
        for j in range(i + 1, _N):
            if np.linalg.norm(best[i] - best[j]) < 1e-5:
                best[j] += [1e-5, 0]

    return [[float(x), float(y)] for (x, y) in best]
