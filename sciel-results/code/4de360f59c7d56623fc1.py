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

def _diameter(pts):
    pts = np.asarray(pts, dtype=float).reshape(_N, 2)
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    return float(d.max())

def _log_product(pts):
    pts = np.asarray(pts, dtype=float).reshape(_N, 2)
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(d, 1.0)
    tri_idx = np.triu_indices(_N, 1)
    return 2.0 * float(np.sum(np.log(d[tri_idx])))

def _score_from_log_product_and_diameter(lp, dia):
    if dia <= 1e-12:
        return 0.0
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
    pts = np.asarray(pts, dtype=float).reshape(_N, 2)
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Stage 1: L-BFGS-B
    try:
        res = minimize(_objective, pts.ravel(), method='L-BFGS-B', options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-10})
        cand = res.x.reshape(_N, 2)
        s = _score(cand)
        if s > best_score:
            best_pts, best_score = cand, s
    except Exception:
        pass

    # Stage 2: SLSQP
    if _remaining() > 5.0:
        try:
            res = minimize(_objective, best_pts.ravel(), method='SLSQP', options={'maxiter': 300, 'ftol': 1e-14})
            cand = res.x.reshape(_N, 2)
            s = _score(cand)
            if s > best_score:
                best_pts, best_score = cand, s
        except Exception:
            pass

    # Stage 3: High precision L-BFGS-B
    if _remaining() > 3.0:
        try:
            res = minimize(_objective, best_pts.ravel(), method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12})
            cand = res.x.reshape(_N, 2)
            s = _score(cand)
            if s > best_score:
                best_pts, best_score = cand, s
        except Exception:
            pass

    return best_pts, best_score

def _two_squares_score(r_outer, r_inner, delta_angle):
    """Score for two concentric squares with given parameters."""
    outer_angles = np.array([0.0, math.pi/2, math.pi, 3*math.pi/2])
    inner_angles = outer_angles + delta_angle
    
    pts_outer = np.column_stack([r_outer * np.cos(outer_angles), r_outer * np.sin(outer_angles)])
    pts_inner = np.column_stack([r_inner * np.cos(inner_angles), r_inner * np.sin(inner_angles)])
    pts = np.vstack([pts_outer, pts_inner])
    return _score(pts)

def _two_squares_objective(params):
    """Objective for optimizing two squares parameters."""
    r_outer, r_inner, delta_angle = params
    if r_outer < 0.01 or r_inner < 0.0:
        return 1e12
    return -_two_squares_score(r_outer, r_inner, delta_angle)

def _radial_objective(radii):
    """Objective for radial deformation with fixed angles."""
    radii = np.asarray(radii, dtype=float)
    thetas = 2 * math.pi * np.arange(_N) / _N
    pts = np.column_stack([radii * np.cos(thetas), radii * np.sin(thetas)])
    return _objective(pts)

def _full_radial_objective(params):
    """Objective for full radial + angular deformation."""
    # params: 8 radii + 8 angles (but angles are constrained to be ordered)
    # Actually, let's use 8 radii and 8 angle offsets from regular
    radii = params[:_N]
    dangles = params[_N:]
    thetas_base = 2 * math.pi * np.arange(_N) / _N
    thetas = thetas_base + dangles
    pts = np.column_stack([radii * np.cos(thetas), radii * np.sin(thetas)])
    return _objective(pts)

def construct():
    t0 = time.time()
    rng = np.random.default_rng(42)
    
    best = None
    best_score = 0.0
    
    # 1. Try regular octagon
    oct_pts = np.array([[math.cos(2 * math.pi * k / _N),
                         math.sin(2 * math.pi * k / _N)] for k in range(_N)], dtype=float)
    cand, s = _refine(oct_pts)
    best, best_score = cand, s

    # 2. Radial deformations (fixed angles, variable radii)
    if _remaining() > 10.0:
        r0 = np.ones(_N)
        bounds = [(0.1, 2.0)] * _N
        try:
            res_r = minimize(_radial_objective, r0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 800, 'ftol': 1e-12})
            radii_opt = res_r.x
            thetas = 2 * math.pi * np.arange(_N) / _N
            pts_radial = np.column_stack([radii_opt * np.cos(thetas), radii_opt * np.sin(thetas)])
            cand, s = _refine(pts_radial)
            if s > best_score:
                best, best_score = cand, s
        except Exception:
            pass

    # 3. Full radial + angular deformation
    if _remaining() > 15.0:
        params0 = np.concatenate([np.ones(_N), np.zeros(_N)])
        bounds_r = [(0.1, 2.0)] * _N
        bounds_a = [(-math.pi/4, math.pi/4)] * _N
        bounds = bounds_r + bounds_a
        try:
            res_fa = minimize(_full_radial_objective, params0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 800, 'ftol': 1e-12})
            radii_opt = res_fa.x[:_N]
            dangles_opt = res_fa.x[_N:]
            thetas_base = 2 * math.pi * np.arange(_N) / _N
            thetas = thetas_base + dangles_opt
            pts_fa = np.column_stack([radii_opt * np.cos(thetas), radii_opt * np.sin(thetas)])
            cand, s = _refine(pts_fa)
            if s > best_score:
                best, best_score = cand, s
        except Exception:
            pass

    # 4. Basin Hopping on the full 16D space, starting from best
    # Increased iterations to utilize time saved by removing specific parametric searches
    if _remaining() > 25.0:
        def bh_objective(x):
            return _objective(x.reshape(_N, 2))
        
        try:
            bh_res = basinhopping(
                bh_objective, best.ravel(), 
                minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 300, 'ftol': 1e-12}},
                niter=200, seed=12345, 
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

    # 5. Random restarts if time remains
    # This explores the landscape more broadly than fixed cages
    while _remaining() > 20.0:
        # Random points in unit disk
        pts_rand = rng.uniform(-1.0, 1.0, size=(8, 2))
        cand, s = _refine(pts_rand)
        if s > best_score:
            best, best_score = cand, s

    # 6. Perturbation-based multi-start from best
    if _remaining() > 20.0:
        for _ in range(5):
            if _remaining() < 15.0:
                break
            noise = rng.normal(0, 0.3, size=(8, 2))
            pts_pert = best + noise
            cand, s = _refine(pts_pert)
            if s > best_score:
                best, best_score = cand, s

    # Final Intensive Polish
    # We spend the last few seconds on very high precision local optimization
    # to ensure we are at the exact peak of the local optimum.
    if _remaining() > 5.0:
        # Stage 1: SLSQP with high precision
        try:
            res_final = minimize(_objective, best.ravel(), method='SLSQP',
                                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12})
            cand = res_final.x.reshape(_N, 2)
            s = _score(cand)
            if s > best_score:
                best, best_score = cand, s
        except Exception:
            pass

    if _remaining() > 3.0:
        # Stage 2: L-BFGS-B with very high precision
        try:
            res_final2 = minimize(_objective, best.ravel(), method='L-BFGS-B',
                                options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14})
            cand = res_final2.x.reshape(_N, 2)
            s = _score(cand)
            if s > best_score:
                best, best_score = cand, s
        except Exception:
            pass
            
    if _remaining() > 1.0:
        # Stage 3: One more SLSQP pass to polish any residual error
        try:
            res_final3 = minimize(_objective, best.ravel(), method='SLSQP',
                                options={'maxiter': 1000, 'ftol': 1e-15})
            cand = res_final3.x.reshape(_N, 2)
            s = _score(cand)
            if s > best_score:
                best, best_score = cand, s
        except Exception:
            pass

    best = np.asarray(best, dtype=float).reshape(_N, 2)
    
    # Ensure distinct points
    for i in range(_N):
        for j in range(i+1, _N):
            if np.linalg.norm(best[i] - best[j]) < 1e-5:
                best[j] += [1e-5, 0]
                
    return [[float(x), float(y)] for (x, y) in best]
