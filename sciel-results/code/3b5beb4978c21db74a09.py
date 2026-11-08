"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, basin_hopping
from scipy.spatial import ConvexHull
import itertools

_start_time = time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill

def _triangle_areas_vec(points):
    """Compute all C(n,3) triangle areas for n points using vectorization."""
    n = len(points)
    if n < 3:
        return np.array([])
    
    # Generate indices for all combinations i < j < k
    # Using itertools is safer for memory if n is large, but n=14 is small.
    # C(14,3) = 364. Very small.
    areas = []
    for i, j, k in itertools.combinations(range(n), 3):
        # Area = 0.5 * |cross product|
        v1 = points[j] - points[i]
        v2 = points[k] - points[i]
        area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
        areas.append(area)
    return np.array(areas)

def _hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    areas = _triangle_areas_vec(points)
    if len(areas) == 0:
        return 0.0
    min_area = np.min(areas)
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    return min_area / hull_area

def _neg_score(points):
    """Negative score for minimization."""
    return -_score(points)

def _normalize_unit_area(points):
    """Normalize points so that their convex hull has unit area."""
    hull_area = _hull_area(points)
    if hull_area > 1e-10:
        points = points / np.sqrt(hull_area)
    return points

def _initial_regular_polygon():
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _initial_concentric_rings():
    # 7 points on outer, 7 on inner
    angles1 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    ring1 = np.column_stack([np.cos(angles1), np.sin(angles1)])
    angles2 = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 7
    ring2 = 0.5 * np.column_stack([np.cos(angles2), np.sin(angles2)])
    return np.vstack([ring1, ring2])

def _initial_hexagonal():
    # Generate a hexagonal lattice and pick 14 points
    # This is a heuristic initialization
    pts = []
    # Center
    pts.append([0, 0])
    # Ring 1: 6 points
    for i in range(6):
        angle = i * np.pi / 3
        pts.append([np.cos(angle), np.sin(angle)])
    # Ring 2: 6 points (rotated)
    for i in range(6):
        angle = i * np.pi / 3 + np.pi / 6
        pts.append([2 * np.cos(angle), 2 * np.sin(angle)])
    # We have 13 points, add one more close to center or on ring 1
    pts.append([0.5, 0.5])
    pts = np.array(pts[:N])
    return pts

def _initial_random(seed):
    rng = np.random.RandomState(seed)
    pts = rng.uniform(-1, 1, (N, 2))
    return _normalize_unit_area(pts)

def construct():
    global _start_time
    best_points = None
    best_score = -1e10
    
    # List of initializations
    inits = [
        _initial_regular_polygon(),
        _initial_concentric_rings(),
        _initial_hexagonal(),
        _initial_random(1),
        _initial_random(2),
        _initial_random(3),
        _initial_random(4),
        _initial_random(5)
    ]
    
    # Optimize each initialization
    for init in inits:
        if time.time() > _DEADLINE - 10:
            break
            
        try:
            # Use Nelder-Mead for non-smooth objective
            result = minimize(
                _neg_score,
                init,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            if result.success or result.fun < 0:
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_points = result.x.copy()
                    
        except Exception:
            continue
            
        # Try Powell as well if time permits
        if time.time() > _DEADLINE - 20:
            continue
            
        try:
            result2 = minimize(
                _neg_score,
                result.x if result.success else init,
                method='Powell',
                options={'maxiter': 200, 'xtol': 1e-9, 'ftol': 1e-9}
            )
            
            if result2.success or result2.fun < 0:
                score2 = -result2.fun
                if score2 > best_score:
                    best_score = score2
                    best_points = result2.x.copy()
                    
        except Exception:
            continue

    # If time allows, try Basin-Hopping on the best point found
    if best_points is not None and time.time() < _DEADLINE - 30:
        try:
            # Define a wrapper for basin_hopping
            def bh_obj(x):
                return _neg_score(x)
            
            bh_result = basin_hopping(
                bh_obj,
                best_points,
                niter=20,
                T=1.0,
                stepsize=0.5,
                minimizer_kwargs={'method': 'Nelder-Mead', 'options': {'maxiter': 200}},
                seed=42
            )
            
            if bh_result.fun < best_score: # Note: bh_result.fun is the minimized value (negative score)
                score_bh = -bh_result.fun
                if score_bh > best_score:
                    best_score = score_bh
                    best_points = bh_result.x.copy()
        except Exception:
            pass

    if best_points is None:
        # Fallback
        best_points = _initial_regular_polygon()
        
    return best_points
# EVOLVE-BLOCK-END
