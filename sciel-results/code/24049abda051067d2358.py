"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, Bounds
from itertools import combinations

_start_time = time.time()
_TIME_LIMIT = 140.0  # safety margin before 150s kill

def _triangle_area(p, q, r):
    """Area of triangle with vertices p, q, r (2D points)."""
    return 0.5 * abs((q[0]-p[0])*(r[1]-p[1]) - (r[0]-p[0])*(q[1]-p[1]))

def _min_triangle_area(pts):
    """Minimum triangle area over all triples of points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = _triangle_area(pts[i], pts[j], pts[k])
                if a < min_area:
                    min_area = a
    return min_area

def _convex_hull_area(pts):
    """Area of convex hull of points using shoelace formula after sorting by angle from centroid."""
    n = len(pts)
    cx = np.mean(pts[:, 0])
    cy = np.mean(pts[:, 1])
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    order = np.argsort(angles)
    hull_pts = pts[order]
    # Shoelace formula
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += hull_pts[i, 0] * hull_pts[j, 1]
        area -= hull_pts[j, 0] * hull_pts[i, 1]
    return 0.5 * abs(area)

def _score(pts):
    """Score = min triangle area / convex hull area."""
    min_ta = _min_triangle_area(pts)
    hull_a = _convex_hull_area(pts)
    if hull_a < 1e-12:
        return 0.0
    return min_ta / hull_a

def _generate_multi_shell(m, k, r_inner, phase):
    """Generate points: m points on outer unit circle, k points on inner circle of radius r_inner, rotated by phase."""
    outer_angles = 2 * np.pi * np.arange(m) / m
    inner_angles = 2 * np.pi * np.arange(k) / k + phase
    
    outer_pts = np.column_stack((np.cos(outer_angles), np.sin(outer_angles)))
    inner_pts = np.column_stack((r_inner * np.cos(inner_angles), r_inner * np.sin(inner_angles)))
    
    return np.vstack([outer_pts, inner_pts])

def _neg_score_multi_shell(params, m, k):
    """Negative score for minimization. params = [r_inner, phase]."""
    r_inner = params[0]
    phase = params[1]
    if r_inner <= 0 or r_inner >= 1:
        return 1e10
    pts = _generate_multi_shell(m, k, r_inner, phase)
    return -_score(pts)

def _refine_points(pts, time_budget=30.0):
    """Refine individual point coordinates using Nelder-Mead on all 2N coordinates."""
    def _neg_score_flat(x):
        p = x.reshape(-1, 2)
        return -_score(p)
    
    x0 = pts.flatten()
    result = minimize(_neg_score_flat, x0, method='Nelder-Mead',
                      options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
    return result.x.reshape(-1, 2)

def construct():
    global _start_time
    elapsed = time.time() - _start_time
    remaining = _TIME_LIMIT - elapsed
    if remaining < 5:
        # Not enough time, return regular 13-gon
        angles = 2 * np.pi * np.arange(N) / N
        return np.column_stack((np.cos(angles), np.sin(angles)))
    
    best_pts = None
    best_score = -1.0
    
    # Try different partitions (m, k) with m + k = 13
    partitions = [(7, 6), (8, 5), (9, 4), (10, 3), (11, 2), (12, 1)]
    
    for m, k in partitions:
        if time.time() - _start_time > _TIME_LIMIT - 20:
            break
        
        # Optimize (r_inner, phase) with Nelder-Mead
        best_params = None
        best_val = 1e10
        
        # Try multiple starting points for robustness
        for r0 in [0.3, 0.5, 0.7]:
            for ph0 in [0.0, np.pi / max(k, 1), np.pi / max(m, 1)]:
                x0 = [r0, ph0]
                try:
                    res = minimize(_neg_score_multi_shell, x0, args=(m, k),
                                   method='Nelder-Mead',
                                   options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True})
                    if res.fun < best_val:
                        best_val = res.fun
                        best_params = res.x
                except Exception:
                    pass
        
        if best_params is not None:
            pts = _generate_multi_shell(m, k, best_params[0], best_params[1])
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
    
    if best_pts is None:
        # Fallback to regular 13-gon
        angles = 2 * np.pi * np.arange(N) / N
        best_pts = np.column_stack((np.cos(angles), np.sin(angles)))
        best_score = _score(best_pts)
    
    # Refine individual points if time allows
    if time.time() - _start_time < _TIME_LIMIT - 30:
        try:
            refined = _refine_points(best_pts, time_budget=25.0)
            refined_score = _score(refined)
            if refined_score > best_score:
                best_pts = refined
                best_score = refined_score
        except Exception:
            pass
    
    # Final check
    if time.time() - _start_time > _TIME_LIMIT - 5:
        pass  # Just return what we have
    
    return best_pts
# EVOLVE-BLOCK-END
