"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import itertools

def _min_triangle_area(points):
    """Calculate the minimum area of any triangle formed by 3 points."""
    min_area = float('inf')
    n = len(points)
    # For N=13, there are C(13,3) = 286 triangles. This is fast.
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * |det(p2-p1, p3-p1)|
                v1 = p2 - p1
                v2 = p3 - p1
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Calculate the area of the convex hull of the points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume # In 2D, .volume is the area
    except Exception:
        return 0.0

def _score(points):
    """Calculate the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    min_tri_area = _min_triangle_area(points)
    return min_tri_area / hull_area

def construct():
    # Initialize with a regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.column_stack((np.cos(angles), np.sin(angles))).astype(float)
    
    best_points = points.copy()
    best_score = _score(best_points)
    
    # Local search / Hill Climbing
    # We will try to improve the score by perturbing points.
    # Since the function is non-smooth (min of areas), simple gradient doesn't work directly.
    # We use a stochastic local search.
    
    step_size = 0.1
    no_improvement_count = 0
    max_no_improvement = 50
    max_iterations = 1000
    
    for it in range(max_iterations):
        if no_improvement_count > max_no_improvement:
            break
            
        improved = False
        # Try perturbing each point
        for i in range(N):
            # Try a few random directions
            for _ in range(10):
                direction = np.random.randn(2)
                direction /= np.linalg.norm(direction)
                new_points = best_points.copy()
                new_points[i] += direction * step_size
                
                new_score = _score(new_points)
                if new_score > best_score:
                    best_points = new_points
                    best_score = new_score
                    improved = True
                    break # Move to next point if this one improved
                    
        if improved:
            no_improvement_count = 0
            # Slowly decrease step size
            step_size *= 0.95
        else:
            no_improvement_count += 1
            # If stuck, try a larger random jump (restart-like)
            if no_improvement_count % 10 == 0:
                best_points = best_points + np.random.randn(N, 2) * step_size
                best_score = _score(best_points)
                if best_score < _score(np.column_stack((np.cos(angles), np.sin(angles)))):
                     # Fallback to regular polygon if we diverged too much
                     best_points = np.column_stack((np.cos(angles), np.sin(angles))).astype(float)
                     best_score = _score(best_points)
                     step_size = 0.1
                     no_improvement_count = 0

    return best_points
# EVOLVE-BLOCK-END
