"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
import sys
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13
NUM_TRIPLES = N * (N - 1) * (N - 2) // 6

# Precompute triple indices
_triple_indices = np.array(list(combinations(range(N), 3)))

# EVOLVE-BLOCK-START

def _compute_hull_area(pts):
    """Compute the area of the convex hull of the points."""
    try:
        # ConvexHull requires at least 3 non-collinear points
        if len(pts) < 3:
            return 0.0
        hull = ConvexHull(pts)
        area = hull.volume
        if area < 1e-9:
            return 0.0
        return area
    except Exception:
        return 0.0

def _compute_min_ratio(pts):
    """
    Compute the minimum triangle area divided by the convex hull area.
    Returns the score (higher is better) or 0 if invalid.
    """
    hull_area = _compute_hull_area(pts)
    if hull_area == 0.0:
        return 0.0
    
    i = _triple_indices[:, 0]
    j = _triple_indices[:, 1]
    k = _triple_indices[:, 2]
    
    p_i = pts[i]
    p_j = pts[j]
    p_k = pts[k]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    return min_area / hull_area

def _smooth_min(ratios, tau):
    """
    Smooth approximation of the minimum using log-sum-exp.
    min(x) ≈ -1/tau * log(sum(exp(-tau * x)))
    """
    min_r = np.min(ratios)
    shifted = -tau * (ratios - min_r)
    # Clip to avoid overflow
    shifted = np.clip(shifted, -50, 0)
    max_shifted = np.max(shifted)
    log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    return min_r - (1.0 / tau) * log_sum

def _objective(params, tau):
    """
    Objective function: minimize -smooth_min(ratios).
    params: 26 values representing p_1 ... p_13.
    We optimize all points freely, but fix p_0 to (0,0) and p_1 to (1,0) 
    to break translational and rotational invariance and scale invariance.
    Actually, fixing p_0 and p_1 fixes translation, rotation, and scale.
    So we only optimize p_2 ... p_12 (11 points * 2 coords = 22 params).
    """
    pts = np.empty((N, 2))
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 0.0]
    pts[2:] = params.reshape(N - 2, 2)
    
    hull_area = _compute_hull_area(pts)
    if hull_area == 0.0:
        return 1e9
    
    i = _triple_indices[:, 0]
    j = _triple_indices[:, 1]
    k = _triple_indices[:, 2]
    
    p_i = pts[i]
    p_j = pts[j]
    p_k = pts[k]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    ratios = areas / hull_area
    approx_min = _smooth_min(ratios, tau)
    return -approx_min

def _exact_objective(params):
    """
    Objective function for Nelder-Mead: minimize -min_ratio.
    """
    pts = np.empty((N, 2))
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 0.0]
    pts[2:] = params.reshape(N - 2, 2)
    
    score = _compute_min_ratio(pts)
    return -score

def _generate_initial_params(rng, method):
    """
    Generate initial parameters for the 22 free coordinates (p_2 ... p_12).
    """
    if method == 'polygon':
        # Regular 13-gon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        # Perturb angles slightly
        angles += rng.uniform(-0.05, 0.05, N)
        # Perturb radii slightly
        radii = 1.0 + rng.normal(0, 0.02, N)
        
        pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        
        # Transform: p[0] -> (0,0), p[1] -> (1,0)
        pts = pts - pts[0]
        v = pts[1]
        length = np.linalg.norm(v)
        if length < 1e-9:
            return rng.uniform(-1, 1, 22)
        pts = pts / length
        theta = np.arctan2(pts[1, 1], pts[1, 0])
        c, s = np.cos(-theta), np.sin(-theta)
        R = np.array([[c, -s], [s, c]])
        pts = pts @ R.T
        return pts[2:].flatten()
    
    elif method == 'halton':
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r
        
        pts = np.zeros((N, 2))
        for i in range(N):
            pts[i, 0] = halton(i + 1, 2)
            pts[i, 1] = halton(i + 1, 3)
        
        # Center and scale
        pts -= np.mean(pts, axis=0)
        max_dist = np.max(np.linalg.norm(pts, axis=1))
        if max_dist > 0:
            pts /= max_dist
        pts *= 0.8 # Keep inside unit circle
        
        # Add small random perturbation
        pts += rng.normal(0, 0.01, (N, 2))
        
        # Transform
        pts = pts - pts[0]
        v = pts[1]
        length = np.linalg.norm(v)
        if length < 1e-9:
            return rng.uniform(-1, 1, 22)
        pts = pts / length
        theta = np.arctan2(pts[1, 1], pts[1, 0])
        c, s = np.cos(-theta), np.sin(-theta)
        R = np.array([[c, -s], [s, c]])
        pts = pts @ R.T
        return pts[2:].flatten()
        
    elif method == 'rings':
        # 1 center, 6 ring 1, 6 ring 2
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        
        angles1 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        r1 = 0.5
        pts[1:7] = np.column_stack((r1 * np.cos(angles1), r1 * np.sin(angles1)))
        
        angles2 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi/6
        r2 = 1.0
        pts[7:13] = np.column_stack((r2 * np.cos(angles2), r2 * np.sin(angles2)))
        
        # Perturb
        pts += rng.normal(0, 0.05, (N, 2))
        
        # Transform
        pts = pts - pts[0]
        v = pts[1]
        length = np.linalg.norm(v)
        if length < 1e-9:
            return rng.uniform(-1, 1, 22)
        pts = pts / length
        theta = np.arctan2(pts[1, 1], pts[1, 0])
        c, s = np.cos(-theta), np.sin(-theta)
        R = np.array([[c, -s], [s, c]])
        pts = pts @ R.T
        return pts[2:].flatten()
        
    else:
        return rng.uniform(-1, 1, 22)

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    best_params = None
    best_score = -np.inf
    
    # Phase 1: Generate diverse starting points
    num_starts = 20
    starts = []
    start_methods = ['polygon', 'halton', 'rings', 'random', 'polygon', 'halton', 'rings', 'random',
                     'polygon', 'halton', 'rings', 'random', 'polygon', 'halton', 'rings', 'random',
                     'polygon', 'halton', 'rings', 'random']
    
    for i in range(num_starts):
        if (time.time() - start_time) > time_limit - 30.0:
            break
        method = start_methods[i % len(start_methods)]
        if method == 'random':
            p = rng.uniform(-1, 1, 22)
        else:
            p = _generate_initial_params(rng, method)
        starts.append(p)
        
        # Initial evaluation
        pts = np.empty((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = p.reshape(N - 2, 2)
        score = _compute_min_ratio(pts)
        if score > best_score:
            best_score = score
            best_params = p.copy()
    
    # Phase 2: Local optimization with L-BFGS-B using smooth min
    # Use multiple tau values for annealing
    tau_schedule = [20, 50, 100, 300, 800]
    
    for tau in tau_schedule:
        if (time.time() - start_time) > time_limit - 15.0:
            break
        
        for i in range(len(starts)):
            if (time.time() - start_time) > time_limit - 10.0:
                break
            
            try:
                res = minimize(
                    _objective,
                    starts[i],
                    args=(tau,),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                
                # Check if improvement
                old_obj = _objective(starts[i], tau)
                new_obj = res.fun
                if new_obj < old_obj:
                    starts[i] = res.x.copy()
                
                # Evaluate true score
                pts = np.empty((N, 2))
                pts[0] = [0.0, 0.0]
                pts[1] = [1.0, 0.0]
                pts[2:] = starts[i].reshape(N - 2, 2)
                score = _compute_min_ratio(pts)
                
                if score > best_score:
                    best_score = score
                    best_params = starts[i].copy()
                    
            except Exception:
                pass
    
    # Phase 3: Final high-precision polish on the best solution
    if best_params is not None and (time.time() - start_time) < time_limit - 10.0:
        # Use Nelder-Mead on the exact objective for the best few candidates
        # Sort starts by score
        start_scores = []
        for p in starts:
            pts = np.empty((N, 2))
            pts[0] = [0.0, 0.0]
            pts[1] = [1.0, 0.0]
            pts[2:] = p.reshape(N - 2, 2)
            start_scores.append(_compute_min_ratio(pts))
        
        sorted_indices = np.argsort(start_scores)[::-1]
        
        # Polish top 5
        for idx in sorted_indices[:5]:
            if (time.time() - start_time) > time_limit - 5.0:
                break
            
            try:
                res_nm = minimize(
                    _exact_objective,
                    starts[idx],
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                
                if res_nm.fun < _exact_objective(starts[idx]):
                    starts[idx] = res_nm.x.copy()
                    pts = np.empty((N, 2))
                    pts[0] = [0.0, 0.0]
                    pts[1] = [1.0, 0.0]
                    pts[2:] = starts[idx].reshape(N - 2, 2)
                    score = _compute_min_ratio(pts)
                    if score > best_score:
                        best_score = score
                        best_params = starts[idx].copy()
            except Exception:
                pass
    
    # Construct final points
    if best_params is None:
        best_params = _generate_initial_params(rng, 'polygon')
    
    final_pts = np.empty((N, 2))
    final_pts[0] = [0.0, 0.0]
    final_pts[1] = [1.0, 0.0]
    final_pts[2:] = best_params.reshape(N - 2, 2)
    
    return final_pts

# EVOLVE-BLOCK-END
