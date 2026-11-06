import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

_N = 8
_DEADLINE = 148.0
_SAFETY = 5.0
_START = time.time()

def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _START)

def _log_score(pts):
    """
    Calculate the log of the score for a set of points.
    Score = Delta / n^n, where Delta is the product of distances of all ordered pairs
    after scaling the set so that the diameter is 2.
    
    Maximizing Score is equivalent to maximizing:
    log_score = sum_{i!=j} log(d'_ij) - N * log(N)
    where d'_ij = d_ij * (2 / D_max)
    
    log_score = sum_{i!=j} (log(d_ij) + log(2) - log(D_max)) - N * log(N)
    = sum_{i!=j} log(d_ij) + N(N-1)(log(2) - log(D_max)) - N * log(N)
    """
    pts = np.asarray(pts, dtype=float).reshape(_N, 2)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.linalg.norm(diff, axis=2)
    
    dia = d.max()
    if dia < 1e-12:
        return -1e10
    
    d_safe = d.copy()
    np.fill_diagonal(d_safe, 1.0)
    
    log_prod = np.log(d_safe).sum()
    log_score = log_prod + _N * (_N - 1) * (math.log(2.0) - math.log(dia)) - _N * math.log(_N)
    
    return log_score

def _score(pts):
    ls = _log_score(pts)
    if ls < -1e9:
        return 0.0
    return math.exp(ls)

def _get_points_from_gaps(gaps):
    """
    Given 8 angular gaps (summing to 2*pi), generate 8 points on the unit circle.
    """
    gaps = np.asarray(gaps, dtype=float)
    total = gaps.sum()
    if total < 1e-12:
        return np.eye(_N, 2)
    gaps = gaps * (2.0 * math.pi / total)
    
    angles = np.zeros(_N)
    angles[0] = 0.0
    for i in range(1, _N):
        angles[i] = angles[i-1] + gaps[i-1]
    
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    return pts

def _objective_circular(gaps):
    """
    Objective for circular optimization: minimize -log_score.
    """
    if _remaining() < 2:
        return 1e10
    
    all_gaps = np.zeros(_N)
    all_gaps[:7] = gaps
    all_gaps[7] = 2.0 * math.pi - np.sum(gaps)
    
    if np.any(all_gaps < 1e-8):
        return 1e10
    
    pts = _get_points_from_gaps(all_gaps)
    
    d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
    np.fill_diagonal(d, 1e12)
    if d.min() < 1e-6:
        return 1e10
        
    return -_log_score(pts)

def _get_points_D2(params):
    """
    Generate 8 points with D2 symmetry.
    params: [a, b, c, d]
    Points: (±a, ±b) and (±c, ±d)
    """
    a, b, c, d = params
    p1 = np.array([a, b])
    p2 = np.array([-a, b])
    p3 = np.array([-a, -b])
    p4 = np.array([a, -b])
    p5 = np.array([c, d])
    p6 = np.array([-c, d])
    p7 = np.array([-c, -d])
    p8 = np.array([c, -d])
    return np.array([p1, p2, p3, p4, p5, p6, p7, p8])

def _objective_D2(params):
    pts = _get_points_D2(params)
    d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
    np.fill_diagonal(d, 1e12)
    if d.min() < 1e-6:
        return 1e12
    return -_log_score(pts)

def _objective_generic_fixed(x):
    """
    Objective for generic optimization with P0=(0,0) and P1=(1,0) fixed.
    x: 14 floats representing P2...P7
    """
    if _remaining() < 2:
        return 1e10
    
    pts = np.zeros((_N, 2))
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 0.0]
    pts[2:] = x.reshape(6, 2)
    
    d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
    np.fill_diagonal(d, 1e12)
    if d.min() < 1e-6:
        return 1e12
        
    return -_log_score(pts)

def _refine_generic(pts, maxiter=2000, ftol=1e-14):
    """
    Refine a full 8-point configuration using L-BFGS-B.
    """
    def obj(x):
        p = x.reshape(_N, 2)
        d = np.linalg.norm(p[:, np.newaxis, :] - p[np.newaxis, :, :], axis=2)
        np.fill_diagonal(d, 1e12)
        if d.min() < 1e-6:
            return 1e12
        return -_log_score(p)
        
    try:
        res = minimize(obj, pts.ravel(), method='L-BFGS-B', 
                       options={'maxiter': maxiter, 'ftol': ftol, 'gtol': 1e-12})
        cand = res.x.reshape(_N, 2)
        d = np.linalg.norm(cand[:, np.newaxis, :] - cand[np.newaxis, :, :], axis=2)
        np.fill_diagonal(d, 1e12)
        if d.min() < 1e-6:
            return pts, _score(pts)
        return cand, _score(cand)
    except Exception:
        return pts, _score(pts)

def construct():
    t0 = time.time()
    best_pts = None
    best_score = 0.0
    
    rng = np.random.default_rng(12345)
    
    # ============================================================
    # PHASE 1: Circular gap optimization (7D) using DE
    # ============================================================
    
    if _remaining() > 30:
        bounds_circ = [(0.1, 2.0 * math.pi)] * 7
        try:
            res_de_circ = differential_evolution(
                _objective_circular, bounds_circ, 
                seed=42, 
                maxiter=100, 
                tol=1e-8,
                polish=True,
                popsize=15
            )
            if res_de_circ.fun < 1e9:
                all_gaps = np.zeros(_N)
                all_gaps[:7] = res_de_circ.x
                all_gaps[7] = 2.0 * math.pi - np.sum(res_de_circ.x)
                
                if np.all(all_gaps > 1e-8):
                    pts = _get_points_from_gaps(all_gaps)
                    s = _score(pts)
                    if s > best_score:
                        best_score = s
                        best_pts = pts.copy()
        except Exception:
            pass
    
    # ============================================================
    # PHASE 2: D2 symmetric optimization (4D) using DE
    # ============================================================
    
    if _remaining() > 30:
        bounds_d2 = [(0.1, 10.0), (0.1, 10.0), (0.1, 10.0), (0.1, 10.0)]
        try:
            res_de_d2 = differential_evolution(
                _objective_D2, bounds_d2, 
                seed=43, 
                maxiter=100, 
                tol=1e-8,
                polish=True,
                popsize=15
            )
            if res_de_d2.fun < 1e10:
                pts = _get_points_D2(res_de_d2.x)
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_pts = pts.copy()
        except Exception:
            pass
    
    # ============================================================
    # PHASE 3: Generic refinement from best found so far
    # ============================================================
    
    candidates = []
    if best_pts is not None:
        candidates.append(best_pts)
    
    # Add regular octagon
    reg_oct = np.array([[math.cos(2*math.pi*k/_N), math.sin(2*math.pi*k/_N)] for k in range(_N)])
    candidates.append(reg_oct)
    
    # Add some random starts
    for _ in range(10):
        if _remaining() < 20:
            break
        # Random points in a square
        pts = rng.uniform(-5, 5, size=(8, 2))
        # Ensure distinct
        d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        np.fill_diagonal(d, 1e12)
        if d.min() > 0.1:
            candidates.append(pts)

    # Refine all candidates
    for pts in candidates:
        if _remaining() < 15:
            break
        cand, s = _refine_generic(pts, maxiter=1000)
        if s > best_score:
            best_score = s
            best_pts = cand
    
    # ============================================================
    # PHASE 4: Fixed-2-Points Generic Optimization (14D)
    # This is more robust for finding global-ish maxima in generic space
    # ============================================================
    
    if _remaining() > 20:
        # Initialize from best_pts by fixing P0, P1
        # We need to transform best_pts such that P0=0, P1=1,0
        # This is a rigid transformation + scaling
        if best_pts is not None:
            p0 = best_pts[0]
            p1 = best_pts[1]
            vec = p1 - p0
            norm_vec = np.linalg.norm(vec)
            if norm_vec > 1e-6:
                # Rotate and translate
                angle = math.atan2(vec[1], vec[0])
                rot = np.array([[math.cos(-angle), -math.sin(-angle)], [math.sin(-angle), math.cos(-angle)]])
                scaled_pts = (best_pts - p0) / norm_vec
                rotated_pts = scaled_pts @ rot.T
                # Now P0 is (0,0), P1 is (1,0)
                x_init = rotated_pts[2:].ravel()
                
                try:
                    res_fixed = minimize(_objective_generic_fixed, x_init, method='L-BFGS-B',
                                         options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12})
                    x_opt = res_fixed.x
                    pts_opt = np.zeros((8, 2))
                    pts_opt[0] = [0, 0]
                    pts_opt[1] = [1, 0]
                    pts_opt[2:] = x_opt.reshape(6, 2)
                    
                    s = _score(pts_opt)
                    if s > best_score:
                        best_score = s
                        best_pts = pts_opt
                except Exception:
                    pass
        
        # Try a few random initializations for the fixed-2-points optimization
        for _ in range(5):
            if _remaining() < 10:
                break
            x_rand = rng.uniform(-5, 5, size=14)
            try:
                res_fixed = minimize(_objective_generic_fixed, x_rand, method='L-BFGS-B',
                                     options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
                x_opt = res_fixed.x
                pts_opt = np.zeros((8, 2))
                pts_opt[0] = [0, 0]
                pts_opt[1] = [1, 0]
                pts_opt[2:] = x_opt.reshape(6, 2)
                
                s = _score(pts_opt)
                if s > best_score:
                    best_score = s
                    best_pts = pts_opt
            except Exception:
                pass

    # ============================================================
    # PHASE 5: Final Polish
    # ============================================================
    
    if best_pts is not None and _remaining() > 10:
        cand, s = _refine_generic(best_pts, maxiter=5000, ftol=1e-15)
        if s > best_score:
            best_score = s
            best_pts = cand

    # Fallback
    if best_pts is None:
        best_pts = np.array([[math.cos(2*math.pi*k/_N), math.sin(2*math.pi*k/_N)] for k in range(_N)])
        best_score = _score(best_pts)

    # Cleanup
    best_pts = np.asarray(best_pts, dtype=float).reshape(_N, 2)
    d = np.linalg.norm(best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :], axis=2)
    np.fill_diagonal(d, 1e12)
    if d.min() < 1e-5:
        for i in range(_N):
            for j in range(i+1, _N):
                if np.linalg.norm(best_pts[i] - best_pts[j]) < 1e-5:
                    best_pts[j] += [1e-5, 0]
    
    max_coord = np.max(np.abs(best_pts))
    if max_coord > 1e6:
        best_pts = best_pts / max_coord * 1e6

    return [[float(x), float(y)] for (x, y) in best_pts]
