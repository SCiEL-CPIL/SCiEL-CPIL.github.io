import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
import itertools

N = 14

# EVOLVE-BLOCK-START

def _min_triangle_area(points):
    """
    Calculate the minimum area of any triangle formed by 3 points in the set.
    Points: (N, 2) array.
    """
    min_area = float('inf')
    n = len(points)
    # Vectorized calculation for all triples is O(N^3) which is 14^3 = 2744, very fast.
    # We can loop over i, j and vectorize over k for better performance if needed, 
    # but for N=14, a simple triple loop or itertools is fine.
    # Using itertools for clarity and sufficient speed.
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            # Vectorized over k > j
            # Area = 0.5 * |cross(pj - pi, pk - pi)|
            # Let v1 = pj - pi
            # Let V2 = points[k:] - pi
            # cross = v1[0]*V2[1] - v1[1]*V2[0]
            v1 = pj - pi
            V2 = points[j+1:] - pi
            if len(V2) == 0:
                continue
            # Cross product in 2D
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
        # Check for degenerate hull
        if np.allclose(points, points[0]):
            return 0.0
        
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
        
        if hull_area < 1e-12:
            return 0.0
            
        min_tri_area = _min_triangle_area(points)
        
        if min_tri_area < 1e-12:
            return 0.0
            
        return min_tri_area / hull_area
    except Exception:
        return 0.0

def _negative_score(points):
    """
    Objective function for minimizer.
    """
    return -_score(points)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    # Initial guess: Regular 14-gon inscribed in unit circle.
    # Area of regular n-gon with radius R is 0.5 * n * R^2 * sin(2*pi/n)
    # Min triangle area for regular n-gon:
    # The smallest triangles are formed by 3 consecutive vertices.
    # Area = 0.5 * R^2 * sin(2*pi/n)
    # Ratio = (0.5 * R^2 * sin(2*pi/n)) / (0.5 * n * R^2 * sin(2*pi/n)) = 1/n
    # For N=14, ratio = 1/14 ≈ 0.0714.
    # This is a strong baseline.
    
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Scale to unit area? 
    # The score is normalized by hull area, so scaling doesn't change the score.
    # However, numerical stability might benefit from keeping coordinates in a reasonable range.
    # Unit circle is fine.
    
    best_points = x0
    best_score = _score(best_points)
    
    # Optimization
    # Nelder-Mead is robust for non-smooth, non-convex problems like this.
    # We will run it for a fixed number of iterations or until time limit.
    
    try:
        # Use a tight initial simplex relative to the scale
        # Default Nelder-Mead options might be too slow or too loose.
        # We'll let it run but monitor time.
        
        # Since the function is invariant to rotation and translation (and scaling cancels out),
        # there is a lot of symmetry. 
        # To help the optimizer, we can fix the first point at (0,0) and the second on the x-axis?
        # No, Nelder-Mead works in the full space. 
        # However, to reduce dimensionality and avoid trivial moves, 
        # we can fix the first point at (0,0) and the second at (1,0) and optimize the rest?
        # This reduces dimensions from 28 to 26.
        # Let's stick to full space for simplicity first, but Nelder-Mead might wander.
        
        # Alternative: Use a specific initialization and local search.
        # Let's try Nelder-Mead on the full 2D coordinates.
        
        res = minimize(_negative_score, x0, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        
        if res.success or res.fun < -best_score:
            candidate_points = res.x
            candidate_score = -res.fun
            if candidate_score > best_score:
                best_points = candidate_points
                best_score = candidate_score
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 5:
        return best_points
    
    # Try a few random restarts with local optimization if time permits
    # The regular polygon is a local minimum of the "maximize min area" problem?
    # Actually, for regular polygons, the min triangle is the one with consecutive vertices.
    # Perturbing vertices might allow increasing the min triangle area by "spreading" them out more evenly?
    # For N=14, 14 is not a multiple of 3, so the "evenly spread" configuration is the regular polygon.
    # However, the convex hull area also changes.
    
    # Let's try a few random perturbations of the regular polygon and optimize them.
    rng = np.random.default_rng(42)
    
    for i in range(10): # Max 10 restarts
        if time.time() > deadline - 10:
            break
            
        # Perturb the regular polygon
        noise = rng.normal(0, 0.1, size=(N, 2))
        x_init = x0 + noise
        
        try:
            res = minimize(_negative_score, x_init, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            
            if res.success or res.fun < -best_score:
                candidate_points = res.x
                candidate_score = -res.fun
                if candidate_score > best_score:
                    best_points = candidate_points
                    best_score = candidate_score
        except Exception:
            continue
            
    return best_points

# EVOLVE-BLOCK-END
