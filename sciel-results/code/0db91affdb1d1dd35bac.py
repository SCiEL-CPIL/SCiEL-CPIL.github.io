import time
import math
import numpy as np
from scipy.optimize import minimize, basinhopping, differential_evolution

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


def _refine(pts, max_iters_stage1=500, max_iters_stage2=200, max_iters_stage3=1000):
    """
    Multi-stage local refinement to polish a candidate configuration.
    1. L-BFGS-B with standard tolerances.
    2. SLSQP for potential constraint-like behavior improvements.
    3. Final high-precision L-BFGS-B.
    """
    pts = np.asarray(pts, dtype=float).reshape(_N, 2)
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Stage 1: Standard L-BFGS-B
    try:
        res = minimize(_objective, pts.ravel(), method='L-BFGS-B', options={'maxiter': max_iters_stage1, 'ftol': 1e-10})
        cand = res.x.reshape(_N, 2)
        s = _score(cand)
        if s > best_score:
            best_pts, best_score = cand, s
    except Exception:
        pass

    # Stage 2: SLSQP
    if _remaining() > 5.0:
        try:
            res = minimize(_objective, best_pts.ravel(), method='SLSQP', options={'maxiter': max_iters_stage2, 'ftol': 1e-12})
            cand = res.x.reshape(_N, 2)
            s = _score(cand)
            if s > best_score:
                best_pts, best_score = cand, s
        except Exception:
            pass

    # Stage 3: High-precision polish
    if _remaining() > 2.0:
        try:
            res = minimize(_objective, best_pts.ravel(), method='L-BFGS-B', options={'maxiter': max_iters_stage3, 'ftol': 1e-14, 'gtol': 1e-12})
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
    # kind 3: Random
    return rng.uniform(-1.0, 1.0, size=(8, 2))


def construct():
    t0 = time.time()
    rng = np.random.default_rng(12345)

    # 1. Seed with regular octagon and refine
    oct_pts = np.array([[math.cos(2 * math.pi * k / _N),
                         math.sin(2 * math.pi * k / _N)] for k in range(_N)], dtype=float)
    best, best_score = _refine(oct_pts)

    # 2. Refine diverse combinatorial topologies
    for kind in range(1, 4):
        if _remaining() < 5.0:
            break
        init = _make_cage_init(kind, rng)
        cand, s = _refine(init)
        if s > best_score:
            best, best_score = cand, s

    # 3. Differential Evolution: Robust global search
    # DE is often more robust than basin hopping for high-dimensional continuous spaces
    # and can find better global minima. We run it with a reasonable number of generations.
    if _remaining() > 30.0:
        def de_objective(x):
            return _objective(x.reshape(_N, 2))
        
        # Define bounds for DE. The scale doesn't matter for the score (invariant),
        # but we need finite bounds. [-2, 2] is a reasonable range for diameter ~2.
        bounds = [(-2.0, 2.0)] * (2 * _N)
        
        try:
            de_res = differential_evolution(
                de_objective, 
                bounds, 
                seed=12345, 
                maxiter=100, 
                popsize=15,
                tol=1e-8,
                init='sobol'
            )
            cand = de_res.x.reshape(_N, 2)
            cand, s = _refine(cand)
            if s > best_score:
                best, best_score = cand, s
        except Exception:
            pass

    # 4. Basin Hopping: Secondary global search strategy
    # Run basin hopping from the current best, and potentially from other seeds if time permits
    if _remaining() > 20.0:
        def bh_objective(x):
            return _objective(x.reshape(_N, 2))
        
        try:
            bh_res = basinhopping(
                bh_objective, best.ravel(), 
                minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 200, 'ftol': 1e-10}},
                niter=50, seed=12345, 
                stepsize=0.5,
                T=100.0,
                accept_test=lambda x, fx, x0, fx0: fx < fx0
            )
            cand = bh_res.x.reshape(_N, 2)
            cand, s = _refine(cand)
            if s > best_score:
                best, best_score = cand, s
        except Exception:
            pass
        
        # If time remains, run basin hopping from a different seed
        if _remaining() > 20.0:
            try:
                init_sq = _make_cage_init(0, rng)
                bh_res2 = basinhopping(
                    bh_objective, init_sq.ravel(), 
                    minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 200, 'ftol': 1e-10}},
                    niter=50, seed=99999, 
                    stepsize=0.5,
                    T=100.0,
                    accept_test=lambda x, fx, x0, fx0: fx < fx0
                )
                cand = bh_res2.x.reshape(_N, 2)
                cand, s = _refine(cand)
                if s > best_score:
                    best, best_score = cand, s
            except Exception:
                pass

    # 5. Extra multi-start random + local refinement while budget remains
    while _remaining() > 10.0:
        init = _make_cage_init(3, rng)
        cand, s = _refine(init)
        if s > best_score:
            best, best_score = cand, s

    # Final high-precision refinement of the best configuration
    if _remaining() > 2.0:
        cand, s = _refine(best, max_iters_stage1=1000, max_iters_stage2=500, max_iters_stage3=2000)
        if s > best_score:
            best, best_score = cand, s

    best = np.asarray(best, dtype=float).reshape(_N, 2)
    
    # Ensure distinct points and bounded coordinates
    for i in range(_N):
        for j in range(i+1, _N):
            if np.linalg.norm(best[i] - best[j]) < 1e-5:
                best[j] += [1e-5, 0]
                
    # Final clamp to ensure |coordinate| < 10^6 (though unlikely to be an issue)
    best = np.clip(best, -1e6, 1e6)
    
    return [[float(x), float(y)] for (x, y) in best]
