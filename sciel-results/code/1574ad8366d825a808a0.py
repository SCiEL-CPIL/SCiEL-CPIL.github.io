import time
import math
import numpy as np
from scipy.optimize import minimize, basinhopping

_N = 8
_DEADLINE = 150.0
_SAFETY = 10.0
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

def _d2_pts(params):
    """D2-symmetric: 4 base points, each reflected through origin.
    params: 8 floats = 4 points (x,y) each."""
    p = np.asarray(params, dtype=float).reshape(4, 2)
    return np.vstack([p, -p])

def _d2_objective(params):
    pts = _d2_pts(params)
    return _objective(pts)

def _two_squares_pts(r_out, r_in, delta):
    """Two concentric squares: outer at radius r_out, inner at r_in,
    with relative rotation delta. D4v symmetric if delta=0 or pi/4."""
    a_out = np.array([0.0, math.pi/2, math.pi, 3*math.pi/2])
    a_in = a_out + delta
    pts_out = np.column_stack([r_out * np.cos(a_out), r_out * np.sin(a_out)])
    pts_in = np.column_stack([r_in * np.cos(a_in), r_in * np.sin(a_in)])
    return np.vstack([pts_out, pts_in])

def _two_squares_objective(params):
    r_out, r_in, delta = params
    if r_out < 0.01 or r_in < 0.0:
        return 1e12
    pts = _two_squares_pts(r_out, r_in, delta)
    return _objective(pts)

def _refine_full(pts):
    """Multi-stage full 16D refinement."""
    pts = np.asarray(pts, dtype=float).reshape(_N, 2)
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Stage 1: L-BFGS-B
    try:
        res = minimize(_objective, best_pts.ravel(), method='L-BFGS-B',
                       options={'maxiter': 300, 'ftol': 1e-10})
        cand = res.x.reshape(_N, 2)
        s = _score(cand)
        if s > best_score:
            best_pts, best_score = cand, s
    except Exception:
        pass

    # Stage 2: SLSQP
    if _remaining() > 3.0:
        try:
            res = minimize(_objective, best_pts.ravel(), method='SLSQP',
                           options={'maxiter': 200, 'ftol': 1e-12})
            cand = res.x.reshape(_N, 2)
            s = _score(cand)
            if s > best_score:
                best_pts, best_score = cand, s
        except Exception:
            pass

    # Stage 3: High precision L-BFGS-B
    if _remaining() > 2.0:
        try:
            res = minimize(_objective, best_pts.ravel(), method='L-BFGS-B',
                           options={'maxiter': 1000, 'ftol': 1e-14})
            cand = res.x.reshape(_N, 2)
            s = _score(cand)
            if s > best_score:
                best_pts, best_score = cand, s
        except Exception:
            pass

    return best_pts, best_score

def _refine_d2(params):
    """Refine in D2-symmetric 8D space."""
    params = np.asarray(params, dtype=float).reshape(4, 2)
    best_params = params.copy()
    pts = _d2_pts(params)
    best_score = _score(pts)
    
    # L-BFGS-B in 8D
    try:
        res = minimize(_d2_objective, best_params.ravel(), method='L-BFGS-B',
                       options={'maxiter': 500, 'ftol': 1e-10})
        cand_params = res.x.reshape(4, 2)
        cand_pts = _d2_pts(cand_params)
        s = _score(cand_pts)
        if s > best_score:
            best_params, best_score = cand_params, s
    except Exception:
        pass

    # SLSQP in 8D
    if _remaining() > 3.0:
        try:
            res = minimize(_d2_objective, best_params.ravel(), method='SLSQP',
                           options={'maxiter': 300, 'ftol': 1e-12})
            cand_params = res.x.reshape(4, 2)
            cand_pts = _d2_pts(cand_params)
            s = _score(cand_pts)
            if s > best_score:
                best_params, best_score = cand_params, s
        except Exception:
            pass

    return _d2_pts(best_params), best_score

def _optimize_two_squares():
    """Optimize the two-squares family (3D parameter space)."""
    best_score = 0.0
    best_pts = None
    best_params = None
    
    # Multiple starting points
    starts = [
        [1.0, 0.5, 0.0],       # regular octagon (delta=0, equal radii would be octagon)
        [1.0, 0.5, math.pi/4], # delta=pi/4
        [1.0, 0.3, 0.0],
        [1.0, 0.7, math.pi/4],
        [1.0, 0.4, math.pi/8],
        [1.0, 0.6, math.pi/4],
        [1.2, 0.5, 0.0],
        [0.8, 0.5, math.pi/4],
        [1.0, 0.2, 0.0],
        [1.0, 0.8, math.pi/4],
    ]
    
    for s in starts:
        if _remaining() < 2.0:
            break
        try:
            res = minimize(_two_squares_objective, s, method='L-BFGS-B',
                           bounds=[(0.05, 3.0), (0.0, 3.0), (-math.pi, math.pi)],
                           options={'maxiter': 500, 'ftol': 1e-10})
            params = res.x
            pts = _two_squares_pts(params[0], params[1], params[2])
            sc = _score(pts)
            if sc > best_score:
                best_score = sc
                best_pts = pts
                best_params = params
        except Exception:
            pass
    
    # Basin hopping in 3D
    if _remaining() > 5.0 and best_params is not None:
        try:
            bh_res = basinhopping(
                _two_squares_objective, best_params,
                minimizer_kwargs={'method': 'L-BFGS-B',
                                  'bounds': [(0.05, 3.0), (0.0, 3.0), (-math.pi, math.pi)],
                                  'options': {'maxiter': 200, 'ftol': 1e-10}},
                niter=50, seed=99, stepsize=0.3, T=50.0,
                accept_test=lambda x, fx, x0, fx0: fx < fx0
            )
            params = bh_res.x
            pts = _two_squares_pts(params[0], params[1], params[2])
            sc = _score(pts)
            if sc > best_score:
                best_score = sc
                best_pts = pts
        except Exception:
            pass
    
    return best_pts, best_score

def construct():
    t0 = time.time()
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_score = 0.0
    
    # 1. Regular octagon
    oct_pts = np.array([[math.cos(2 * math.pi * k / _N),
                         math.sin(2 * math.pi * k / _N)] for k in range(_N)], dtype=float)
    cand, s = _refine_full(oct_pts)
    best_pts, best_score = cand, s
    
    # 2. Two-squares optimization (3D)
    if _remaining() > 15.0:
        cand, s = _optimize_two_squares()
        if cand is not None and s > best_score:
            best_pts, best_score = cand, s
            # Full 16D refinement from two-squares result
            if _remaining() > 5.0:
                cand2, s2 = _refine_full(cand)
                if s2 > best_score:
                    best_pts, best_score = cand2, s2
    
    # 3. D2-symmetric optimization from various starts
    if _remaining() > 15.0:
        d2_starts = [
            # Regular octagon in D2 form (4 points, reflected)
            np.array([[1.0, 0.0], [math.cos(math.pi/4), math.sin(math.pi/4)],
                      [0.0, 1.0], [math.cos(math.pi/4), -math.sin(math.pi/4)]])[:4],
            # Slightly deformed
            np.array([[1.0, 0.0], [0.7, 0.7], [0.0, 1.0], [0.7, -0.7]]),
            np.array([[1.0, 0.0], [0.8, 0.5], [0.0, 1.0], [0.5, -0.8]]),
            np.array([[1.0, 0.1], [0.7, 0.7], [0.0, 1.0], [0.6, -0.6]]),
            # From best so far, extract D2 part
        ]
        
        # Also try D2 from current best
        if best_pts is not None:
            # Project to D2: average with negative
            bp = np.asarray(best_pts).reshape(_N, 2)
            # Try to find 4 points such that -p_i gives the other 4
            # Just use first 4 as approximation
            d2_starts.append(bp[:4].copy())
        
        for ds in d2_starts:
            if _remaining() < 3.0:
                break
            ds = np.asarray(ds, dtype=float).reshape(4, 2)
            cand, s = _refine_d2(ds)
            if s > best_score:
                best_pts, best_score = cand, s
                # Full refinement
                if _remaining() > 3.0:
                    cand2, s2 = _refine_full(cand)
                    if s2 > best_score:
                        best_pts, best_score = cand2, s2
    
    # 4. Basin hopping in full 16D from best
    if _remaining() > 20.0:
        def bh_obj(x):
            return _objective(x.reshape(_N, 2))
        try:
            bh_res = basinhopping(
                bh_obj, best_pts.ravel(),
                minimizer_kwargs={'method': 'L-BFGS-B',
                                  'options': {'maxiter': 150, 'ftol': 1e-10}},
                niter=80, seed=777, stepsize=0.4, T=80.0,
                accept_test=lambda x, fx, x0, fx0: fx < fx0
            )
            cand = bh_res.x.reshape(_N, 2)
            cand, s = _refine_full(cand)
            if s > best_score:
                best_pts, best_score = cand, s
        except Exception:
            pass
    
    # 5. More structured initializations
    if _remaining() > 10.0:
        # Concentric circles: 4 outer, 4 inner at different angles
        for r_in in [0.3, 0.5, 0.7]:
            for rot in [0.0, math.pi/8, math.pi/4]:
                if _remaining() < 3.0:
                    break
                pts_c = []
                for k in range(4):
                    a = 2*math.pi*k/4
                    pts_c.append([math.cos(a), math.sin(a)])
                for k in range(4):
                    a = 2*math.pi*k/4 + rot
                    pts_c.append([r_in*math.cos(a), r_in*math.sin(a)])
                pts_c = np.array(pts_c, dtype=float)
                cand, s = _refine_full(pts_c)
                if s > best_score:
                    best_pts, best_score = cand, s
    
    # 6. Random restarts
    while _remaining() > 8.0:
        pts_rand = rng.uniform(-1.0, 1.0, size=(8, 2))
        cand, s = _refine_full(pts_rand)
        if s > best_score:
            best_pts, best_score = cand, s
    
    # Final polish
    if _remaining() > 2.0:
        cand, s = _refine_full(best_pts)
        if s > best_score:
            best_pts, best_score = cand, s
    
    best_pts = np.asarray(best_pts, dtype=float).reshape(_N, 2)
    
    # Ensure distinct points
    for i in range(_N):
        for j in range(i+1, _N):
            if np.linalg.norm(best_pts[i] - best_pts[j]) < 1e-6:
                best_pts[j] += [1e-6, 0]
    
    return [[float(x), float(y)] for (x, y) in best_pts]
