import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools

N = 14

# EVOLVE-BLOCK-START

def _min_triangle_area_fast(points):
    """
    Calculate the minimum area of any triangle formed by 3 points in the set.
    Points: (N, 2) array.
    Vectorized for speed.
    """
    min_area = float('inf')
    n = len(points)
    # We can vectorize the calculation for all pairs (i, j) and then find min over k.
    # However, a full O(N^3) vectorization creates an N^3 array which is 2744 elements, very small.
    # Let's just do it efficiently.
    
    # Generate indices
    # For each i, j, k with i < j < k
    # Area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
    
    # A simple loop over i and j, vectorizing k is often faster than full vectorization 
    # due to memory overhead of creating large intermediate arrays for N=14? 
    # Actually, 14^3 is tiny. Let's try a fully vectorized approach using broadcasting if possible, 
    # or just the loop which is very fast in Python for N=14.
    
    # Let's use the loop over i, j and vectorize k for balance of speed and simplicity.
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            v1 = pj - pi
            # Vectorize over k > j
            V2 = points[j+1:] - pi
            if len(V2) == 0:
                continue
            # Cross product in 2D: v1[0]*v2[1] - v1[1]*v2[0]
            cross_vals = v1[0] * V2[:, 1] - v1[1] * V2[:, 0]
            areas = 0.5 * np.abs(cross_vals)
            if len(areas) > 0:
                current_min = np.min(areas)
                if current_min < min_area:
                    min_area = current_min
                    # Early exit if we hit 0? Not really useful for optimization.
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
            
        min_tri_area = _min_triangle_area_fast(points)
        
        if min_tri_area < 1e-12:
            return 0.0
            
        return min_tri_area / hull_area
    except Exception:
        return 0.0

def _apply_constraints(points):
    """
    Reduce dimensionality and remove symmetries.
    1. Translate so centroid is at (0,0).
    2. Rotate so first point is on the positive x-axis (y=0).
    This removes 3 degrees of freedom (2 translation, 1 rotation).
    """
    # 1. Center
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    
    # 2. Rotate
    # Angle of first point
    p0 = points_centered[0]
    angle = np.arctan2(p0[1], p0[0])
    
    # Rotation matrix
    c, s = np.cos(-angle), np.sin(-angle)
    R = np.array([[c, -s], [s, c]])
    
    points_rotated = points_centered @ R.T
    
    # Due to floating point errors, the first point's y might be ~1e-16.
    # We can force it to 0 to be strict, but for optimization it's better to let the optimizer handle it
    # or just return the rotated points. The score function is invariant to these transforms,
    # so the score of points_rotated is the same as points.
    
    return points_rotated

def _negative_score_constrained(x):
    """
    Objective for optimization.
    x is a 2D array of shape (N, 2) representing the raw coordinates.
    We apply constraints (centering and rotation) before scoring.
    Note: The optimizer will try to move points, but the constraints project them back.
    This is a "penalty" or "projection" approach. 
    Actually, a better way is to optimize in the reduced space directly, 
    but Nelder-Mead works on the full vector. 
    We can just score the constrained version.
    """
    # x is (N, 2)
    # Apply constraints
    pts = _apply_constraints(x)
    return -_score(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    # Initial guess: Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Apply constraints to get a valid starting point for the "constrained" view
    # But we will optimize the raw x0.
    
    best_points = x0
    best_score = _score(_apply_constraints(best_points))
    
    # Simulated Annealing
    # We will perturb the raw coordinates, apply constraints, and score.
    
    current_points = x0.copy()
    current_score = best_score
    
    T = 1.0 # Initial temperature
    cooling_rate = 0.999
    min_T = 1e-6
    
    # We want to run as many steps as possible.
    # Each step involves:
    # 1. Perturbation
    # 2. Constraint application (fast)
    # 3. Scoring (ConvexHull + MinTri)
    
    # Estimate time per step:
    # ConvexHull for 14 points is ~1-2 ms?
    # MinTri is ~0.1 ms?
    # Let's assume ~2ms per step.
    # 145s / 0.002s = 72,500 steps.
    
    rng = np.random.default_rng(42)
    
    step_count = 0
    no_improvement_count = 0
    max_no_improvement = 5000 # Restart if stuck? Or just keep going with lower T.
    
    while time.time() < deadline - 5.0:
        # Generate perturbation
        # Gaussian noise
        # Scale noise by current temperature
        # We need to be careful with the scale. 
        # Points are roughly in [-1, 1].
        # Noise std should start around 0.1 and decrease.
        
        noise_scale = T * 0.5 # Heuristic
        if noise_scale < 1e-5:
            noise_scale = 1e-5
            
        perturbed_points = current_points + rng.normal(0, noise_scale, size=(N, 2))
        
        # Apply constraints
        new_points = _apply_constraints(perturbed_points)
        new_score = _score(new_points)
        
        # Acceptance criterion
        if new_score > current_score:
            current_points = perturbed_points # Keep the raw points for next perturbation? 
            # Actually, it's better to keep the constrained points as the base for next perturbation?
            # If we keep raw points, the constraints will project them again.
            # If we keep constrained points, we are optimizing in the constrained manifold.
            # Let's keep the constrained points as the "current" state for the next step.
            current_points = new_points
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points
                no_improvement_count = 0
            else:
                no_improvement_count += 1
        else:
            # Metropolis criterion
            delta = new_score - current_score
            if delta < 0:
                # Accept with probability exp(delta / T)
                # delta is negative, so delta/T is negative
                # If T is small, this is very small.
                accept_prob = np.exp(delta / T)
                if rng.random() < accept_prob:
                    current_points = new_points
                    current_score = new_score
            # Else reject
            
        # Cool down
        T *= cooling_rate
        
        step_count += 1
        
        # Optional: If stuck for a long time, try a random restart?
        # For now, just let it cool.
        
    # Final check: Return the best points found.
    # Ensure the returned points are valid (constrained)
    return _apply_constraints(best_points)

# EVOLVE-BLOCK-END
