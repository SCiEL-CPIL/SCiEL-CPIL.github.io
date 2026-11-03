"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import sys
from itertools import combinations

N = 13

# EVOLVE-BLOCK-START

def _all_tri_indices():
    """Precompute all C(13,3) triangle index triples."""
    return list(combinations(range(N), 3))

TRI_INDICES = _all_tri_indices()
NUM_TRIS = len(TRI_INDICES)  # 286

def _compute_min_tri_area_norm(pts):
    """
    Compute the minimum area of any triangle formed by pts,
    normalized by the area of the convex hull of pts.
    Returns (min_area_ratio, min_area, hull_area) or (-1e9, 0, 0) if invalid.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return -1e9, 0.0, 0.0
    
    if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
        return -1e9, 0.0, 0.0
    
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area < 1e-12:
            return -1e9, 0.0, 0.0
    except Exception:
        return -1e9, 0.0, 0.0
    
    min_area = float('inf')
    for i, j, k in TRI_INDICES:
        v1 = pts[j] - pts[i]
        v2 = pts[k] - pts[i]
        area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
        if area < min_area:
            min_area = area
    
    if min_area <= 0:
        return -1e9, 0.0, hull_area
    
    ratio = min_area / hull_area
    return ratio, min_area, hull_area

def _all_tri_areas(pts):
    """Compute all 286 triangle areas for a point set."""
    areas = np.empty(NUM_TRIS)
    for idx, (i, j, k) in enumerate(TRI_INDICES):
        v1 = pts[j] - pts[i]
        v2 = pts[k] - pts[i]
        areas[idx] = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
    return areas

def _soft_min_ratio(pts, temperature):
    """
    Smooth surrogate: soft-min of triangle areas / hull area.
    Uses log-sum-exp for numerical stability.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return -1e9
    
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area < 1e-12:
            return -1e9
    except Exception:
        return -1e9
    
    areas = _all_tri_areas(pts)
    areas = np.maximum(areas, 1e-15)
    
    # soft_min = -T * log(sum(exp(-a/T)))
    scaled = -areas / temperature
    scaled_max = np.max(scaled)
    log_sum = np.log(np.sum(np.exp(scaled - scaled_max))) + scaled_max
    soft_min_area = -log_sum * temperature
    
    return soft_min_area / hull_area

def _objective(x, temperature):
    """Objective for L-BFGS-B: minimize negative soft-min ratio."""
    pts = x.reshape(N, 2)
    ratio = _soft_min_ratio(pts, temperature)
    if ratio < -1e8:
        return 1e8
    return -ratio

def _halton_sequence(n, base):
    """Generate n points from the Halton sequence in [0,1)."""
    result = np.zeros(n)
    for i in range(n):
        f = 1.0
        idx = i + 1
        while idx > 0:
            f /= base
            idx = (idx - 1) // base
        result[i] = f
    return result

def _generate_halton_points(n, bases=(2, 3)):
    """Generate n points in [0,1)^2 using Halton sequences."""
    return np.column_stack([_halton_sequence(n, b) for b in bases])

def _regular_polygon(n):
    """Generate n points on a regular polygon."""
    angles = 2 * np.pi * np.arange(n) / n
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _structured_grid(n):
    """Generate n points on a structured grid (sqrt-ish layout)."""
    # For 13 points, use a 4x4 grid with one removed, or 3x5 with 2 removed
    # Better: use a centered layout
    pts = []
    # Center
    pts.append([0.5, 0.5])
    # Ring 1: 4 points
    for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
        pts.append([0.5 + 0.35*np.cos(angle), 0.5 + 0.35*np.sin(angle)])
    # Ring 2: 4 points
    for angle in [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]:
        pts.append([0.5 + 0.45*np.cos(angle), 0.5 + 0.45*np.sin(angle)])
    # Ring 3: 4 points
    for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
        pts.append([0.5 + 0.48*np.cos(angle + np.pi/8), 0.5 + 0.48*np.sin(angle + np.pi/8)])
    return np.array(pts)

def _optimize_from_start(initial_pts, time_budget, rng):
    """
    Multi-phase optimization from a single starting point:
    1. Coarse soft-min optimization with decreasing temperature
    2. Fine polish with very low temperature
    3. Nelder-Mead on exact objective
    """
    start_time = time.time()
    x0 = initial_pts.flatten().copy()
    best_x = x0.copy()
    best_score = _compute_min_tri_area_norm(initial_pts)[0]
    
    # Phase 1: Soft-min optimization with annealing
    temperatures = [0.05, 0.02, 0.01, 0.005, 0.002]
    for temp in temperatures:
        if time.time() - start_time > time_budget - 3:
            break
        
        def obj(x):
            return _objective(x, temp)
        
        try:
            result = minimize(
                obj, x0, method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-9}
            )
            if result.fun < 1e7:
                cand = result.x
                cand_score = _compute_min_tri_area_norm(cand.reshape(N, 2))[0]
                if cand_score > best_score:
                    best_score = cand_score
                    best_x = cand.copy()
                x0 = cand
            else:
                x0 = result.x
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead on exact objective
    if time.time() - start_time < time_budget - 2:
        def exact_obj(x):
            pts = x.reshape(N, 2)
            r, _, _ = _compute_min_tri_area_norm(pts)
            if r < -1e8:
                return 1e8
            return -r
        
        try:
            result = minimize(
                exact_obj, best_x, method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10,
                         'initial_simplex': 1e-3}
            )
            if result.fun < 1e7:
                cand = result.x
                cand_score = _compute_min_tri_area_norm(cand.reshape(N, 2))[0]
                if cand_score > best_score:
                    best_score = cand_score
                    best_x = cand.copy()
        except Exception:
            pass
    
    # Phase 3: Local perturbation search on the best found
    if time.time() - start_time < time_budget - 1:
        best_pts = best_x.reshape(N, 2)
        for _ in range(50):
            if time.time() - start_time > time_budget - 0.5:
                break
            # Perturb the two points of the minimum triangle
            areas = _all_tri_areas(best_pts)
            min_idx = np.argmin(areas)
            i, j, k = TRI_INDICES[min_idx]
            
            # Try moving one of the three points slightly
            for pidx in [i, j, k]:
                for scale in [0.01, 0.005, 0.002]:
                    if time.time() - start_time > time_budget - 0.3:
                        break
                    trial = best_pts.copy()
                    trial[pidx] += rng.normal(0, scale, size=2)
                    trial_score, _, _ = _compute_min_tri_area_norm(trial)
                    if trial_score > best_score:
                        best_score = trial_score
                        best_x = trial.flatten().copy()
                        best_pts = trial
                        break
    
    return best_x.reshape(N, 2), best_score

def construct():
    """Main entry point: find optimal 13-point configuration."""
    start_time = time.time()
    total_budget = 145.0  # Safety margin for 150s limit
    
    rng = np.random.RandomState(12345)
    
    best_points = None
    best_ratio = -1e9
    
    # Generate diverse starting points
    starts = []
    
    # 1. Halton (2,3)
    starts.append(_generate_halton_points(N, (2, 3)))
    # 2. Halton (3,5)
    starts.append(_generate_halton_points(N, (3, 5)))
    # 3. Halton (5,7)
    starts.append(_generate_halton_points(N, (5, 7)))
    # 4. Regular 13-gon
    starts.append(_regular_polygon(N))
    # 5. Structured concentric rings
    starts.append(_structured_grid(N))
    # 6. Halton with random perturbation
    h = _generate_halton_points(N, (2, 3))
    starts.append(h + rng.normal(0, 0.08, size=(N, 2)))
    # 7. Random uniform
    starts.append(rng.uniform(0.05, 0.95, size=(N, 2)))
    # 8. Halton rotated
    h = _generate_halton_points(N, (2, 3))
    theta = 0.7
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    starts.append(h @ rot.T)
    # 9. Two concentric rings (6+7)
    inner_angles = 2*np.pi*np.arange(6)/6
    outer_angles = 2*np.pi*np.arange(7)/7 + np.pi/7
    inner = np.column_stack([0.3*np.cos(inner_angles), 0.3*np.sin(inner_angles)])
    outer = np.column_stack([0.48*np.cos(outer_angles), 0.48*np.sin(outer_angles)])
    starts.append(np.vstack([inner, outer]) + 0.5)
    # 10. Fibonacci lattice
    golden = (1 + np.sqrt(5)) / 2
    fib_pts = np.zeros((N, 2))
    for i in range(N):
        fib_pts[i, 0] = ((i + 0.5) / N)
        fib_pts[i, 1] = ((i / golden) % 1.0)
    starts.append(fib_pts)
    
    # Evaluate all starts
    for s in starts:
        s = np.clip(s, 0.01, 0.99)
        r, _, _ = _compute_min_tri_area_norm(s)
        if r > best_ratio:
            best_ratio = r
            best_points = s.copy()
    
    # Optimize from each start, allocating time proportionally
    elapsed = time.time() - start_time
    remaining = total_budget - elapsed
    if remaining < 3:
        return best_points
    
    # Sort starts by initial score (best first) to prioritize good starts
    scored_starts = []
    for s in starts:
        s = np.clip(s, 0.01, 0.99)
        r, _, _ = _compute_min_tri_area_norm(s)
        scored_starts.append((r, s))
    scored_starts.sort(key=lambda x: -x[0])
    
    # Allocate time: give more time to better starts
    # Use a budget allocation: top 3 get 25s, next 3 get 15s, rest get 8s
    time_alloc = []
    for idx in range(len(scored_starts)):
        if idx < 3:
            time_alloc.append(25.0)
        elif idx < 6:
            time_alloc.append(15.0)
        else:
            time_alloc.append(8.0)
    
    for idx, (init_score, init_pts) in enumerate(scored_starts):
        if time.time() - start_time > total_budget - 2:
            break
        
        t_budget = min(time_alloc[idx], total_budget - (time.time() - start_time) - 2)
        if t_budget < 2:
            break
        
        opt_pts, opt_ratio = _optimize_from_start(init_pts, t_budget, rng)
        
        if opt_ratio > best_ratio:
            best_ratio = opt_ratio
            best_points = opt_pts.copy()
    
    # Final: try to polish the best result with a very short Nelder-Mead
    if best_points is not None and time.time() - start_time < total_budget - 3:
        def exact_obj(x):
            pts = x.reshape(N, 2)
            r, _, _ = _compute_min_tri_area_norm(pts)
            if r < -1e8:
                return 1e8
            return -r
        
        try:
            result = minimize(
                exact_obj, best_points.flatten(), method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-11,
                         'initial_simplex': 5e-4}
            )
            if result.fun < 1e7:
                final_ratio, _, _ = _compute_min_tri_area_norm(result.x.reshape(N, 2))
                if final_ratio > best_ratio:
                    best_ratio = final_ratio
                    best_points = result.x.reshape(N, 2)
        except Exception:
            pass
    
    if best_points is None:
        best_points = _generate_halton_points(N, (2, 3))
    
    return best_points

# EVOLVE-BLOCK-END
