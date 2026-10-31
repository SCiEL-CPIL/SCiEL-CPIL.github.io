import numpy as np
import time
import scipy.spatial as spat
from scipy.optimize import differential_evolution, minimize
from itertools import combinations

N = 13
# Pre-compute triangle indices
_triple_idx = list(combinations(range(N), 3))
_i_arr = np.array([t[0] for t in _triple_idx])
_j_arr = np.array([t[1] for t in _triple_idx])
_k_arr = np.array([t[2] for t in _triple_idx])

def _triangle_areas(points):
    """Compute areas of all C(N,3) triangles."""
    pts = points
    v1 = pts[_j_arr] - pts[_i_arr]
    v2 = pts[_k_arr] - pts[_i_arr]
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(det)

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = spat.ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-9

def calculate_score(points):
    """
    Calculate the minimum triangle area divided by the convex hull area.
    """
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area <= 1e-9:
        return 0.0
    areas = _triangle_areas(points)
    min_area = np.min(areas)
    return min_area / hull_area

def _points_from_angles(angles):
    """
    Generate 13 points on the unit circle given angles.
    We assume the convex hull is the polygon formed by these points in angular order.
    To ensure the points are in convex position and form the hull, we sort by angle.
    """
    # Normalize angles to [0, 2*pi)
    angles = np.mod(angles, 2 * np.pi)
    # Sort angles to ensure the polygon is simple and convex (points in angular order)
    # Note: If we just take points on the circle, the convex hull is the polygon connecting them in angular order.
    # The area of the hull is the area of this polygon.
    sorted_angles = np.sort(angles)
    x = np.cos(sorted_angles)
    y = np.sin(sorted_angles)
    return np.column_stack((x, y))

def _score_from_angles(angles):
    """
    Calculate score directly from angles.
    This is the objective to maximize.
    """
    pts = _points_from_angles(angles)
    return calculate_score(pts)

def _negative_score_from_angles(angles):
    """
    Objective for minimizers (minimize negative score).
    """
    return -_score_from_angles(angles)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Strategy:
    # 1. Global search using Differential Evolution on the 13 angular variables.
    #    The domain for each angle is [0, 2*pi).
    #    However, since we sort the angles, the specific assignment of which point is which doesn't matter,
    #    only the set of 13 angles matters.
    #    DE works well on bounded domains.
    
    # Initial guess: Regular polygon
    # angles_i = i * 2*pi / 13
    init_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    # Bounds for DE: [0, 2*pi) for each angle
    bounds = [(0, 2 * np.pi) for _ in range(N)]
    
    best_score = -1.0
    best_angles = init_angles.copy()
    
    # Phase 1: Differential Evolution
    # DE is stochastic and robust to local minima.
    # We limit the number of iterations and function evaluations to stay within time.
    
    # Estimate time per evaluation:
    # _score_from_angles involves sorting (N log N), computing areas (C(N,3) ~ 286 ops), hull (N log N).
    # This is very fast, likely < 1ms per eval.
    # We can afford thousands of evaluations.
    
    # Use DE with specific parameters to balance exploration and exploitation
    # maxiter: max generations
    # popsize: population size multiplier
    # tol: convergence tolerance
    
    try:
        # We will run DE.
        # Note: DE minimizes by default, so we use _negative_score_from_angles
        # But we want to maximize score, so we minimize -score.
        
        # To ensure we don't timeout, we'll wrap the objective with a time check?
        # DE handles the loop. We just set maxiter and popsize.
        # Let's estimate:
        # popsize=15 -> 15*13 = 195 points per generation.
        # maxiter=100 -> 19500 evaluations.
        # If each eval is 0.5ms, that's ~10 seconds.
        # This is safe.
        
        res = differential_evolution(
            _negative_score_from_angles,
            bounds,
            seed=42,
            maxiter=200,
            popsize=20,
            tol=1e-10,
            init='latinhypercube',
            polish=True  # Use L-BFGS-B to polish the result
        )
        
        current_time = time.time()
        if current_time < deadline:
            best_angles = res.x
            best_score = -res.fun
    except Exception as e:
        print(f"DE failed: {e}")
        # Fallback to regular polygon if DE fails completely
        best_angles = init_angles
        best_score = _score_from_angles(best_angles)

    # Phase 2: Local Polish with L-BFGS-B
    # DE's polish might be sufficient, but we can do another pass with a tighter tolerance
    # or a different starting point if we have time.
    
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        try:
            # Use the result from DE as start point
            res2 = minimize(
                _negative_score_from_angles,
                best_angles,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            current_time = time.time()
            if current_time < deadline:
                score2 = -res2.fun
                if score2 > best_score:
                    best_score = score2
                    best_angles = res2.x
        except Exception:
            pass

    # Phase 3: Multi-start Local Optimization
    # Try a few different random initializations to see if we can beat the DE result.
    # Since DE is global, this is less critical, but random restarts of L-BFGS-B can find better local optima.
    
    remaining_time = deadline - time.time()
    if remaining_time > 10.0:
        n_restarts = 5
        time_per_restart = remaining_time / (n_restarts + 1)
        
        for i in range(n_restarts):
            if time.time() > deadline - 2.0:
                break
            
            # Random initialization: perturb regular polygon
            angles_rand = init_angles + 0.1 * np.random.randn(N)
            
            try:
                res_r = minimize(
                    _negative_score_from_angles,
                    angles_rand,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if time.time() > deadline - 2.0:
                    break
                    
                score_r = -res_r.fun
                if score_r > best_score:
                    best_score = score_r
                    best_angles = res_r.x
            except Exception:
                continue

    # Final check
    if time.time() > deadline - 1.0:
        # If we are about to timeout, ensure we return a valid point
        pass

    # Convert best angles to points
    best_points = _points_from_angles(best_angles)
    
    # One final score calculation to be sure
    final_score = calculate_score(best_points)
    
    # Fallback if something went wrong
    if final_score <= 0 or not np.isfinite(final_score):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        final_score = calculate_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    print(calculate_score(pts))
