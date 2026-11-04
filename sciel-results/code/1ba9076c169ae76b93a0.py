"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

N = 13
TRI_INDICES = list(combinations(range(N), 3))
NUM_TRIS = len(TRI_INDICES)  # 286

# Precompute index arrays for vectorized area computation
_I = np.array([t[0] for t in TRI_INDICES])
_J = np.array([t[1] for t in TRI_INDICES])
_K = np.array([t[2] for t in TRI_INDICES])

def _tri_areas(pts):
    """Vectorized triangle areas for all 286 triples."""
    pi = pts[_I]
    pj = pts[_J]
    pk = pts[_K]
    v1 = pj - pi
    v2 = pk - pi
    return 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])

def _hull_area(pts):
    try:
        return ConvexHull(pts).volume
    except Exception:
        return 0.0

def _score(pts):
    """Exact score: min triangle area / convex hull area."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2) or np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
        return 0.0
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 0.0
    return float(np.min(_tri_areas(pts)) / ha)

def _soft_min(areas, tau):
    """
    Compute smooth minimum using log-sum-exp.
    """
    max_area = np.max(areas)
    log_sum_exp = np.log(np.sum(np.exp(-(areas - max_area) / tau)))
    return max_area - tau * log_sum_exp

def _neg_soft_min_ratio(pts, temp):
    """
    Smooth surrogate to maximize: -soft_min(areas) / hull_area.
    """
    pts = np.asarray(pts, dtype=np.float64)
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 1e8
    a = _tri_areas(pts)
    a = np.maximum(a, 1e-15)
    s = -a / temp
    s = s - np.max(s)
    logsum = np.log(np.sum(np.exp(s)))
    soft_min = -logsum * temp
    return soft_min / ha

def _decode_fourier(params):
    """
    Decode Fourier parameters into 2D points.
    params[0] = r0 (base radius)
    params[1:] = Fourier coefficients (a1, b1, a2, b2, ...)
    """
    r0 = params[0]
    coeffs = params[1:]
    
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radii = np.full(N, r0)
    
    # Number of modes is len(coeffs) // 2
    num_modes = len(coeffs) // 2
    for m in range(num_modes):
        a_m = coeffs[2 * m]
        b_m = coeffs[2 * m + 1]
        radii += a_m * np.cos(m * angles) + b_m * np.sin(m * angles)
        
    # Ensure positive radii
    radii = np.maximum(radii, 1e-6)
    
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return pts

def _objective_fourier(params, tau=0.02):
    """
    Objective for Fourier parameterization.
    Minimizes negative smooth score.
    """
    pts = _decode_fourier(params)
    
    # Compute all triangle areas
    n = N
    i, j, k = np.indices((n, n, n))
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    p_i = pts[i]
    p_j = pts[j]
    p_k = pts[k]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    areas = 0.5 * np.abs(cross)
    
    if np.any(areas == 0):
        return 1e10
    
    # Compute hull area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 1e10
        
    if hull_area < 1e-10:
        return 1e10

    # Smooth min using log-sum-exp
    smooth_min = _soft_min(areas, tau)
    score = smooth_min / hull_area
    return -score

def _decode_polar(params):
    """
    Decode polar parameters (log-radii, angles) into 2D points.
    Sorts points by angle to enforce convex order.
    """
    log_radii = params[:N]
    angles = params[N:2*N]
    
    # Ensure angles are in [0, 2*pi)
    angles = np.mod(angles, 2 * np.pi)
    
    # Sort by angle
    sort_idx = np.argsort(angles)
    sorted_angles = angles[sort_idx]
    sorted_log_radii = log_radii[sort_idx]
    
    # Exponentiate to get positive radii
    radii = np.exp(sorted_log_radii)
    
    pts = np.column_stack([
        radii * np.cos(sorted_angles),
        radii * np.sin(sorted_angles)
    ])
    return pts

def _objective_polar(params, tau=0.02):
    """
    Objective for Polar parameterization.
    """
    pts = _decode_polar(params)
    
    # Compute all triangle areas
    n = N
    i, j, k = np.indices((n, n, n))
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    p_i = pts[i]
    p_j = pts[j]
    p_k = pts[k]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    areas = 0.5 * np.abs(cross)
    
    if np.any(areas == 0):
        return 1e10
    
    # Compute hull area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 1e10
        
    if hull_area < 1e-10:
        return 1e10

    # Smooth min using log-sum-exp
    smooth_min = _soft_min(areas, tau)
    score = smooth_min / hull_area
    return -score

def _refine_cartesian(pts, time_budget):
    """
    Refine a Cartesian point set using L-BFGS-B on a smooth surrogate.
    This acts as the local polish step.
    """
    t0 = time.time()
    best = pts.copy()
    best_s = _score(pts)
    
    # Use a moderately small temperature for the soft-min to guide L-BFGS
    # towards the exact min without being too noisy.
    temp = 0.005
    
    def obj(x):
        p = x.reshape(N, 2)
        return _neg_soft_min_ratio(p, temp)
    
    try:
        res = minimize(obj, best.flatten(), method='L-BFGS-B',
                       options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
        cand = res.x.reshape(N, 2)
        s = _score(cand)
        if s > best_s:
            best_s = s
            best = cand.copy()
    except Exception:
        pass
        
    # Second pass with tighter temperature if time permits
    if time.time() - t0 < time_budget - 1:
        temp2 = 0.001
        def obj2(x):
            p = x.reshape(N, 2)
            return _neg_soft_min_ratio(p, temp2)
        try:
            res = minimize(obj2, best.flatten(), method='L-BFGS-B',
                           options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8})
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass
            
    return best, best_s

def construct():
    """
    Main entry: Hybrid Fourier + Polar search with local polish.
    1. Global search in Fourier space (low-dimensional, smooth manifold).
    2. Global search in Polar space (higher dimension, more flexible).
    3. Local polish in Cartesian space.
    """
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin for 120s limit
    
    best_pts = None
    best_score = -np.inf
    
    # --- Phase 1: Fourier Search ---
    # Parameters: r0, a1, b1, a2, b2, a3, b3
    # 7 parameters total.
    # Bounds:
    # r0: [0.1, 2.0]
    # coeffs: [-0.5, 0.5] (to keep convexity likely)
    
    bounds_f = [(0.1, 2.0)] + [(-0.5, 0.5)] * 6
    # Initial guess: regular polygon (r0=1, all coeffs 0)
    x0_f = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Run Differential Evolution in Fourier space
    # DE is robust for global search in low dims
    try:
        # Use a wrapper to enforce deadline
        def de_obj(params):
            if time.time() > deadline - 5.0:
                return 1e10
            return _objective_fourier(params)
            
        # maxiter is hard to control precisely with DE in scipy, 
        # but we can limit popsize and use tol.
        # Actually, scipy DE doesn't have a built-in time limit.
        # We will run a fixed number of iterations or rely on the objective penalty.
        # Better: use a smaller population and fewer iterations, then check time.
        
        # Let's run DE with a modest budget first.
        res_f = differential_evolution(
            de_obj, 
            bounds_f, 
            seed=42,
            maxiter=50,  # Limit iterations
            popsize=15,
            tol=1e-4,
            disp=False
        )
        
        if res_f.fun < -best_score:
            pts_f = _decode_fourier(res_f.x)
            score_f = _score(pts_f)
            if score_f > best_score:
                best_score = score_f
                best_pts = pts_f
    except Exception:
        pass
    
    # --- Phase 2: Polar Search (Multi-start L-BFGS-B) ---
    # If Fourier didn't find a good solution, or to try to beat it,
    # try Polar space.
    
    # We can use the best Fourier solution as a starting point for Polar?
    # It's easier to just multi-start Polar from random regular polygons.
    
    num_polar_starts = 10
    time_per_start = (deadline - time.time()) / num_polar_starts
    
    if time_per_start > 2.0: # Only if we have time
        for i in range(num_polar_starts):
            if time.time() > deadline - 5.0:
                break
            
            # Generate initial polar params
            # Start near regular polygon
            rng = np.random.default_rng(1000 + i)
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.normal(0, 0.1, N)
            log_radii = np.zeros(N) + rng.normal(0, 0.1, N)
            x0_p = np.concatenate([log_radii, angles])
            
            bounds_p = [(-1.0, 1.5)] * N + [(0.0, 2 * np.pi)] * N
            
            try:
                # Use a wrapper to enforce deadline
                def polar_obj(params):
                    if time.time() > deadline - 5.0:
                        return 1e10
                    return _objective_polar(params)
                    
                res_p = minimize(
                    polar_obj,
                    x0_p,
                    method='L-BFGS-B',
                    bounds=bounds_p,
                    options={'maxiter': 100, 'ftol': 1e-10, 'disp': False}
                )
                
                if res_p.fun < -best_score:
                    pts_p = _decode_polar(res_p.x)
                    score_p = _score(pts_p)
                    if score_p > best_score:
                        best_score = score_p
                        best_pts = pts_p
            except Exception:
                pass

    # Fallback if nothing worked
    if best_pts is None:
        th = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(th), np.sin(th)])
        best_score = _score(best_pts)

    # --- Phase 3: Final Polish in Cartesian ---
    remaining = deadline - time.time() - 2
    if remaining > 5:
        try:
            pts_final, score_final = _refine_cartesian(best_pts, remaining)
            if score_final > best_score:
                best_score = score_final
                best_pts = pts_final.copy()
        except Exception:
            pass
            
    return best_pts
