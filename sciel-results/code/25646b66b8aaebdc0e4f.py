"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

# Precompute triangle indices
TRI_INDICES = list(combinations(range(N), 3))
NUM_TRIS = len(TRI_INDICES)  # 286

def _all_tri_areas(pts):
    """Compute all 286 triangle areas for a point set."""
    pts = np.asarray(pts, dtype=np.float64)
    areas = np.empty(NUM_TRIS)
    for idx, (i, j, k) in enumerate(TRI_INDICES):
        v1 = pts[j] - pts[i]
        v2 = pts[k] - pts[i]
        areas[idx] = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
    return areas

def _convex_hull_area(pts):
    """Compute area of convex hull."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return 0.0
    if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
        return 0.0
    areas = _all_tri_areas(pts)
    min_area = np.min(areas)
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-12:
        return 0.0
    return min_area / hull_area

def _smooth_max_min_ratio(pts, temperature):
    """
    Smooth surrogate using log-sum-exp for max-min of triangle areas / hull area.
    This is more numerically stable and allows better gradient flow.
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
    
    # Use smooth min via log-sum-exp: -T * log(sum(exp(-a/T)))
    scaled = -areas / temperature
    scaled_max = np.max(scaled)
    log_sum = np.log(np.sum(np.exp(scaled - scaled_max))) + scaled_max
    smooth_min_area = -log_sum * temperature
    
    return smooth_min_area / hull_area

def _angular_to_cartesian(angles, radii):
    """Convert angular + radial parameters to Cartesian coordinates."""
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y])

def _cartesian_to_angular(pts):
    """Convert Cartesian points to (angle, radius) representation."""
    pts = np.asarray(pts, dtype=np.float64)
    angles = np.arctan2(pts[:, 1], pts[:, 0])
    radii = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
    return angles, radii

def _angular_objective(x, temperature):
    """
    Objective in angular/radial parameterization.
    x = [angles (N), radii (N)]
    """
    angles = x[:N]
    radii = x[N:]
    
    # Ensure radii are positive
    radii = np.maximum(radii, 1e-6)
    
    pts = _angular_to_cartesian(angles, radii)
    ratio = _smooth_max_min_ratio(pts, temperature)
    if ratio < -1e8:
        return 1e8
    return -ratio

def _optimize_angular(initial_pts, time_budget, rng):
    """
    Optimize in angular/radial parameterization with smooth max-min surrogate.
    """
    start_time = time.time()
    angles, radii = _cartesian_to_angular(initial_pts)
    x0 = np.concatenate([angles, radii])
    
    best_x = x0.copy()
    best_score = _score(initial_pts)
    
    # Multi-stage temperature annealing
    temperatures = [0.08, 0.04, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    for temp in temperatures:
        if time.time() - start_time > time_budget - 2:
            break
        
        def obj(x):
            return _angular_objective(x, temp)
        
        try:
            result = minimize(
                obj, x0, method='L-BFGS-B',
                bounds=[(None, None)] * N + [(1e-6, None)] * N,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10,
                         'maxls': 50}
            )
            if result.fun < 1e7:
                cand = result.x
                cand_pts = _angular_to_cartesian(cand[:N], np.maximum(cand[N:], 1e-6))
                cand_score = _score(cand_pts)
                if cand_score > best_score:
                    best_score = cand_score
                    best_x = cand.copy()
                x0 = cand
            else:
                x0 = result.x
        except Exception:
            pass
    
    # Final Nelder-Mead polish on exact objective
    if time.time() - start_time < time_budget - 1:
        def exact_obj(x):
            angles = x[:N]
            radii = np.maximum(x[N:], 1e-6)
            pts = _angular_to_cartesian(angles, radii)
            r = _score(pts)
            return -r
        
        try:
            result = minimize(
                exact_obj, best_x, method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12,
                         'initial_simplex': 1e-3}
            )
            if result.fun < 1e7:
                cand = result.x
                cand_pts = _angular_to_cartesian(cand[:N], np.maximum(cand[N:], 1e-6))
                cand_score = _score(cand_pts)
                if cand_score > best_score:
                    best_score = cand_score
                    best_x = cand.copy()
        except Exception:
            pass
    
    best_pts = _angular_to_cartesian(best_x[:N], np.maximum(best_x[N:], 1e-6))
    return best_pts, best_score

def _optimize_hybrid(initial_pts, time_budget, rng):
    """
    Hybrid optimization: alternate between Cartesian and angular parameterizations.
    """
    start_time = time.time()
    best_pts = initial_pts.copy()
    best_score = _score(initial_pts)
    
    # Phase 1: Angular optimization
    if time.time() - start_time < time_budget - 2:
        try:
            ang_pts, ang_score = _optimize_angular(
                initial_pts, 
                min(time_budget - 2, 30.0), 
                rng
            )
            if ang_score > best_score:
                best_score = ang_score
                best_pts = ang_pts.copy()
        except Exception:
            pass
    
    # Phase 2: Cartesian L-BFGS with smooth surrogate
    if time.time() - start_time < time_budget - 2:
        x0 = best_pts.flatten().copy()
        temperatures = [0.02, 0.01, 0.005, 0.002]
        
        for temp in temperatures:
            if time.time() - start_time > time_budget - 2:
                break
            
            def obj(x):
                pts = x.reshape(N, 2)
                ratio = _smooth_max_min_ratio(pts, temp)
                if ratio < -1e8:
                    return 1e8
                return -ratio
            
            try:
                result = minimize(
                    obj, x0, method='L-BFGS-B',
                    options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if result.fun < 1e7:
                    cand = result.x
                    cand_score = _score(cand.reshape(N, 2))
                    if cand_score > best_score:
                        best_score = cand_score
                        best_pts = cand.reshape(N, 2).copy()
                    x0 = cand
                else:
                    x0 = result.x
            except Exception:
                pass
    
    # Phase 3: Nelder-Mead on exact objective
    if time.time() - start_time < time_budget - 1:
        def exact_obj(x):
            pts = x.reshape(N, 2)
            r = _score(pts)
            return -r
        
        try:
            result = minimize(
                exact_obj, best_pts.flatten(), method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-10, 'fatol': 1e-12,
                         'initial_simplex': 1e-3}
            )
            if result.fun < 1e7:
                cand = result.x
                cand_score = _score(cand.reshape(N, 2))
                if cand_score > best_score:
                    best_score = cand_score
                    best_pts = cand.reshape(N, 2).copy()
        except Exception:
            pass
    
    return best_pts, best_score

def _regular_polygon(n, radius=1.0, center=(0,0)):
    """Generate regular polygon vertices."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([center[0] + radius*np.cos(angles), 
                            center[1] + radius*np.sin(angles)])

def _concentric_rings(n, n_rings=3):
    """Generate points on concentric rings for diverse initialization."""
    pts = []
    # Center point
    pts.append([0.0, 0.0])
    remaining = n - 1
    for ring in range(n_rings):
        if remaining <= 0:
            break
        n_on_ring = min(remaining, max(3, (n-1)//n_rings))
        if ring == n_rings - 1:
            n_on_ring = remaining
        radius = (ring + 1) / n_rings
        angles = np.linspace(0, 2*np.pi, n_on_ring, endpoint=False) + ring * np.pi/6
        ring_pts = np.column_stack([radius*np.cos(angles), radius*np.sin(angles)])
        pts.append(ring_pts)
        remaining -= n_on_ring
    
    if remaining > 0:
        angles = np.linspace(0, 2*np.pi, remaining, endpoint=False)
        ring_pts = np.column_stack([1.0*np.cos(angles), 1.0*np.sin(angles)])
        pts.append(ring_pts)
    
    result = np.vstack(pts)
    return result[:n]

def construct():
    """Main entry point: find optimal 13-point configuration."""
    start_time = time.time()
    total_budget = 145.0  # Safety margin for 150s limit
    
    rng = np.random.RandomState(12345)
    
    best_points = None
    best_ratio = -1e9
    
    # Generate diverse starting points
    starts = []
    
    # 1. Regular 13-gon
    starts.append(_regular_polygon(N, radius=1.0))
    # 2. Concentric rings (3 rings)
    starts.append(_concentric_rings(N, n_rings=3))
    # 3. Concentric rings (4 rings)
    starts.append(_concentric_rings(N, n_rings=4))
    
    # 4-7: Random points in unit disk with good spread
    for _ in range(4):
        angles = rng.uniform(0, 2*np.pi, N)
        angles.sort()
        radii = rng.uniform(0.3, 1.0, N)
        pts = np.column_stack([radii*np.cos(angles), radii*np.sin(angles)])
        starts.append(pts)
    
    # 8: Jittered regular polygon
    base_poly = _regular_polygon(N, radius=1.0)
    jitter = rng.normal(0, 0.05, (N, 2))
    starts.append(base_poly + jitter)
    
    # Evaluate initial candidates
    for cand in starts:
        if time.time() - start_time > total_budget:
            break
        try:
            s = _score(cand)
            if s > best_ratio:
                best_ratio = s
                best_points = cand.copy()
        except Exception:
            pass
    
    # Optimize from each starting point using hybrid method
    n_starts = len(starts)
    time_per_start = (total_budget - 5.0) / n_starts  # 5s safety margin
    
    for i, start in enumerate(starts):
        if time.time() - start_time > total_budget - 2:
            break
        
        remaining_time = total_budget - (time.time() - start_time)
        alloc_time = min(time_per_start, remaining_time - 1.0)
        if alloc_time < 3.0:
            continue
        
        try:
            opt_pts, opt_score = _optimize_hybrid(start, alloc_time, rng)
            if opt_score > best_ratio:
                best_ratio = opt_score
                best_points = opt_pts.copy()
        except Exception:
            pass
    
    # Final refinement on the best found
    if best_points is not None and time.time() - start_time < total_budget - 3:
        final_time = total_budget - (time.time() - start_time) - 1.0
        if final_time > 10.0:
            try:
                final_pts, final_score = _optimize_hybrid(best_points, final_time, rng)
                if final_score > best_ratio:
                    best_ratio = final_score
                    best_points = final_pts.copy()
            except Exception:
                pass
    
    if best_points is None:
        # Fallback: regular polygon
        best_points = _regular_polygon(N, radius=1.0)
    
    return best_points
# EVOLVE-BLOCK-END
