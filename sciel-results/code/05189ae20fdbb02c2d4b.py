"""Heilbronn convex regions: maximize minimum triangle area over convex hull area."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

def _triangle_areas(pts):
    """Compute all C(n,3) triangle areas from n points."""
    n = len(pts)
    areas = np.empty((n * (n - 1) * (n - 2)) // 6)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = pts[i]
                b = pts[j]
                c = pts[k]
                areas[idx] = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
                idx += 1
    return areas

def _convex_hull_area(pts):
    """Compute area of convex hull of points."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    areas = _triangle_areas(pts)
    min_area = np.min(areas)
    hull_area = _convex_hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    return min_area / hull_area

def _negative_score(pts):
    """Negative score for minimization."""
    return -_score(pts)

def _soft_min_objective(pts, temperature=1.0):
    """Smooth approximation of negative log-sum-exp for triangle areas."""
    areas = _triangle_areas(pts)
    hull_area = _convex_hull_area(pts)
    if hull_area <= 1e-12:
        return 1e6
    # Normalize areas by hull area to make scale-invariant
    normalized = areas / hull_area
    # Use log-sum-exp for soft min
    shifted = normalized - np.max(normalized)
    log_sum_exp = np.log(np.sum(np.exp(shifted))) + np.max(normalized)
    # We want to maximize the minimum, so minimize -soft_min
    # soft_min = -log(sum(exp(-a/T))) * T
    # For minimization of the negative: we want large min area
    # Let's use: objective = -soft_min where soft_min approximates min
    # Actually, let's directly minimize the negative of a soft-min
    # soft_min(a) = -T * log(sum(exp(-a_i/T)))
    # We want to maximize soft_min, so minimize -soft_min = T * log(sum(exp(-a_i/T)))
    shifted_neg = -normalized / temperature
    shifted_neg -= np.max(shifted_neg)
    log_sum = np.log(np.sum(np.exp(shifted_neg))) + np.max(shifted_neg)
    return temperature * log_sum

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s
    
    best_score = 0.0
    best_pts = None
    
    # Start with a regular 13-gon inscribed in unit circle
    angles = 2 * np.pi * np.arange(N) / N
    pts0 = np.column_stack([np.cos(angles), np.sin(angles)])
    # Normalize to unit area
    hull_area0 = _convex_hull_area(pts0)
    if hull_area0 > 1e-12:
        pts0 = pts0 * np.sqrt(1.0 / hull_area0)
    
    score0 = _score(pts0)
    if score0 > best_score:
        best_score = score0
        best_pts = pts0.copy()
    
    # Try optimization from multiple starting points
    rng = np.random.default_rng(N)
    
    # Initial random perturbations of the regular polygon
    attempts = 0
    max_attempts = 20
    
    for attempt in range(max_attempts):
        if time.time() > deadline - 5.0:
            break
        
        # Create perturbed initial point
        if attempt == 0:
            pts_init = pts0.copy()
        else:
            # Random perturbation
            angles_pert = 2 * np.pi * np.arange(N) / N + rng.normal(0, 0.1, N)
            radii = 1.0 + rng.normal(0, 0.1, N)
            radii = np.clip(radii, 0.5, 1.5)
            pts_init = np.column_stack([radii * np.cos(angles_pert), radii * np.sin(angles_pert)])
            # Normalize
            ha = _convex_hull_area(pts_init)
            if ha > 1e-12:
                pts_init = pts_init * np.sqrt(1.0 / ha)
        
        # Optimize using L-BFGS-B with smooth objective
        try:
            result = minimize(
                _soft_min_objective,
                pts_init.flatten(),
                args=(0.01,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10},
                timeout=30
            )
            
            if time.time() > deadline - 5.0:
                break
            
            if result.success or result.fun < 1e6:
                pts_opt = result.x.reshape(N, 2)
                # Ensure points are in reasonable range
                if np.all(np.abs(pts_opt) < 10):
                    sc = _score(pts_opt)
                    if sc > best_score:
                        best_score = sc
                        best_pts = pts_opt.copy()
        except Exception:
            pass
        
        attempts += 1
    
    # Local refinement using direct minimization of negative score
    if best_pts is not None and time.time() < deadline - 10.0:
        try:
            result = minimize(
                _negative_score,
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if time.time() < deadline - 5.0:
                pts_refined = result.x.reshape(N, 2)
                if np.all(np.abs(pts_refined) < 10):
                    sc = _score(pts_refined)
                    if sc > best_score:
                        best_score = sc
                        best_pts = pts_refined.copy()
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback: return regular polygon
        return pts0
    
    return best_pts
# EVOLVE-BLOCK-END
