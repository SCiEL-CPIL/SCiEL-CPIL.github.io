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

    # 2. Two concentric squares: Global optimization in 3D parameter space
    if _remaining() > 20.0:
        def two_squares_bh_objective(params):
            r_outer, r_inner, delta_angle = params
            if r_outer < 0.01 or r_inner < 0.0:
                return 1e12
            return _two_squares_objective(params)

        best_3d_score = -1e18
        best_3d_params = None
        
        # Coarse grid to find a good starting point for basinhopping
        r_outer_grid = np.linspace(0.5, 1.5, 8)
        r_inner_grid = np.linspace(0.1, 1.0, 8)
        delta_grid = np.linspace(0.0, math.pi/2, 8)
        
        for ro in r_outer_grid:
            for ri in r_inner_grid:
                for da in delta_grid:
                    s = _two_squares_score(ro, ri, da)
                    if s > best_3d_score:
                        best_3d_score = s
                        best_3d_params = (ro, ri, da)
        
        if best_3d_params is not None and _remaining() > 10.0:
            try:
                # Use differential evolution for global optimization of the 3D space
                bounds = [(0.1, 2.0), (0.01, 1.5), (0.0, math.pi/2)]
                de_res = differential_evolution(
                    _two_squares_objective,
                    bounds,
                    seed=999,
                    maxiter=100,
                    tol=1e-8,
                    polish=True
                )
                ro_opt, ri_opt, da_opt = de_res.x
                
                # Build the 8 points
                outer_angles = np.array([0.0, math.pi/2, math.pi, 3*math.pi/2])
                inner_angles = outer_angles + da_opt
                pts_outer = np.column_stack([ro_opt * np.cos(outer_angles), ro_opt * np.sin(outer_angles)])
                pts_inner = np.column_stack([ri_opt * np.cos(inner_angles), ri_opt * np.sin(inner_angles)])
                pts_3d = np.vstack([pts_outer, pts_inner])
                cand, s = _refine(pts_3d)
                if s > best_score:
                    best, best_score = cand, s
            except Exception:
                pass

    # 3. Basin Hopping on the full 16D space, starting from best
    if _remaining() > 25.0:
        def bh_objective(x):
            return _objective(x.reshape(_N, 2))
        
        try:
            from scipy.optimize import basinhopping
            bh_res = basinhopping(
                bh_objective, best.ravel(), 
                minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 300, 'ftol': 1e-12}},
                niter=120, seed=12345, 
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

    # 4. Multiple diverse cage initializations
    if _remaining() > 30.0:
        # Cage 1: Two squares rotated 45 degrees
        pts_cage1 = []
        for k in range(4):
            ang = 2 * math.pi * k / 4 + math.pi / 8
            pts_cage1.append([1.0 * math.cos(ang), 1.0 * math.sin(ang)])
        for k in range(4):
            ang = 2 * math.pi * k / 4
            pts_cage1.append([0.6 * math.cos(ang), 0.6 * math.sin(ang)])
        pts_cage1 = np.array(pts_cage1, dtype=float)
        cand, s = _refine(pts_cage1)
        if s > best_score:
            best, best_score = cand, s

    if _remaining() > 20.0:
        # Cage 2: Alternating radii
        radii_cage2 = np.array([1.0, 0.7, 1.0, 0.7, 1.0, 0.7, 1.0, 0.7])
        thetas = 2 * math.pi * np.arange(_N) / _N
        pts_cage2 = np.column_stack([radii_cage2 * np.cos(thetas), radii_cage2 * np.sin(thetas)])
        cand, s = _refine(pts_cage2)
        if s > best_score:
            best, best_score = cand, s

    if _remaining() > 15.0:
        # Cage 3: Clustered pairs
        pts_cage3 = []
        for k in range(4):
            ang = 2 * math.pi * k / 4
            pts_cage3.append([1.0 * math.cos(ang), 1.0 * math.sin(ang)])
            pts_cage3.append([0.5 * math.cos(ang + 0.3), 0.5 * math.sin(ang + 0.3)])
        pts_cage3 = np.array(pts_cage3, dtype=float)
        cand, s = _refine(pts_cage3)
        if s > best_score:
            best, best_score = cand, s

    # 5. Random restarts if time remains
    while _remaining() > 15.0:
        # Random points in unit disk
        pts_rand = rng.uniform(-1.0, 1.0, size=(8, 2))
        cand, s = _refine(pts_rand)
        if s > best_score:
            best, best_score = cand, s

    # 6. Perturbation-based multi-start from best
    if _remaining() > 20.0:
        for _ in range(3):
            if _remaining() < 10.0:
                break
            noise = rng.normal(0, 0.3, size=(8, 2))
            pts_pert = best + noise
            cand, s = _refine(pts_pert)
            if s > best_score:
                best, best_score = cand, s

    # Final polish
    if _remaining() > 3.0:
        cand, s = _refine(best)
        if s > best_score:
            best, best_score = cand, s
    
    # Additional high-precision polish on the final best
    if _remaining() > 2.0:
        try:
            res_final = minimize(_objective, best.ravel(), method='SLSQP',
                                options={'maxiter': 1000, 'ftol': 1e-15})
            cand = res_final.x.reshape(_N, 2)
            s = _score(cand)
            if s > best_score:
                best, best_score = cand, s
        except Exception:
            pass
    
    # Ultra-high precision final polish
    if _remaining() > 1.0:
        try:
            res_final2 = minimize(_objective, best.ravel(), method='L-BFGS-B',
                                options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14})
            cand = res_final2.x.reshape(_N, 2)
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
