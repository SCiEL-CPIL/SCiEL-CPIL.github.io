import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools

N = 14

# EVOLVE-BLOCK-START

def _min_triangle_area_fast(points):
    """
    Calculate the minimum area of any triangle formed by 3 points in the set.
    Points: (N, 2) array.
    """
    min_area = float('inf')
    n = len(points)
    
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            v1 = pj - pi
            V2 = points[j+1:] - pi
            if len(V2) == 0:
                continue
            cross_vals = v1[0] * V2[:, 1] - v1[1] * V2[:, 0]
            areas = 0.5 * np.abs(cross_vals)
            if len(areas) > 0:
                current_min = np.min(areas)
                if current_min < min_area:
                    min_area = current_min
    return min_area

def _score(points):
    """
    Score = Min Triangle Area / Convex Hull Area.
    """
    try:
        if np.allclose(points, points[0]):
            return 0.0
        
        hull = ConvexHull(points)
        hull_area = hull.volume
        
        if hull_area < 1e-12:
            return 0.0
            
        min_tri_area = _min_triangle_area_fast(points)
        
        if min_tri_area < 1e-12:
            return 0.0
            
        return min_tri_area / hull_area
    except Exception:
        return 0.0

def _angles_to_points(angles):
    """
    Convert angles to points on the unit circle.
    """
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _negative_score_from_angles(angles):
    """
    Objective function for optimization.
    angles: 1D array of shape (N,)
    Returns negative score.
    """
    # Sort angles to ensure convex position (though order doesn't matter for hull area/min tri,
    # it's good practice for numerical stability in some contexts, though here we just compute directly)
    # Actually, sorting is not strictly necessary for the score calculation, but it helps
    # if we were relying on polygon perimeter etc. For min tri area, order is irrelevant.
    # However, ensuring they are distinct is key.
    
    points = _angles_to_points(angles)
    score = _score(points)
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    # Initial guess: Regular 14-gon
    angles0 = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    best_points = _angles_to_points(angles0)
    best_score = _score(best_points)
    
    rng = np.random.default_rng(42)
    
    # We will perform multiple restarts with different initial conditions
    # and use Powell's method which is effective for non-smooth objectives.
    
    # Strategy:
    # 1. Start with regular polygon.
    # 2. Perturb angles slightly and optimize.
    # 3. Try random starts.
    # 4. Keep the best result.
    
    # Time budget allocation:
    # Each Powell run might take a few seconds depending on iterations.
    # Let's try to fit as many as possible within the deadline.
    
    # Helper to run a single optimization
    def run_optimization(init_angles, maxiter=200):
        try:
            # Powell method
            res = minimize(_negative_score_from_angles, init_angles, method='Powell',
                           options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-10})
            opt_angles = res.x
            # Ensure angles are within [0, 2*pi) for consistency, though not strictly needed for cos/sin
            opt_angles = opt_angles % (2 * np.pi)
            opt_points = _angles_to_points(opt_angles)
            opt_score = _score(opt_points)
            return opt_points, opt_score, res
        except Exception:
            return None, -1, None

    # Phase 1: Optimize from regular polygon
    # We can also try slightly jittered regular polygons
    for i in range(5):
        if time.time() > deadline - 10.0:
            break
            
        if i == 0:
            init_angles = angles0.copy()
        else:
            # Jitter
            init_angles = angles0 + rng.normal(0, 0.1, size=N)
            
        pts, score, res = run_optimization(init_angles, maxiter=300)
        if pts is not None and score > best_score:
            best_score = score
            best_points = pts
            
    # Phase 2: Random restarts
    # Try a few random configurations
    # To make random starts more likely to be good, we can start with a "loose" regular polygon
    # or just random angles.
    # Given the landscape, random angles often lead to degenerate triangles (small area).
    # It's better to start from structured patterns.
    
    # Let's try different "frequencies" or phase shifts
    for i in range(10):
        if time.time() > deadline - 5.0:
            break
            
        # Random permutation of regular polygon angles? No, that's the same set.
        # Random offsets?
        # Let's try a mix of regular and random
        if i % 2 == 0:
            # Jittered regular
            init_angles = angles0 + rng.normal(0, 0.2, size=N)
        else:
            # Random but sorted (to avoid immediate self-intersection issues in thought, though score handles it)
            init_angles = np.sort(rng.uniform(0, 2 * np.pi, N))
            
        pts, score, res = run_optimization(init_angles, maxiter=200)
        if pts is not None and score > best_score:
            best_score = score
            best_points = pts

    # Final local refinement using Nelder-Mead on the best found angles
    # Extract angles from best_points
    best_angles = np.arctan2(best_points[:, 1], best_points[:, 0])
    # Ensure continuity? Not strictly necessary for cos/sin, but for NM it might help to keep them close
    # Let's just run NM on the current angles
    
    if time.time() < deadline - 2.0:
        try:
            res_nm = minimize(_negative_score_from_angles, best_angles, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            nm_angles = res_nm.x % (2 * np.pi)
            nm_points = _angles_to_points(nm_angles)
            nm_score = _score(nm_points)
            if nm_score > best_score:
                best_score = nm_score
                best_points = nm_points
        except Exception:
            pass

    return best_points

# EVOLVE-BLOCK-END
