"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import random

def min_triangle_area(points):
    """Calculate the minimum area of any triangle formed by 3 distinct points."""
    n = len(points)
    min_area = float('inf')
    # Iterate over all triples
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                # Area = 0.5 * |cross(p2-p1, p3-p1)|
                v1 = p2 - p1
                v2 = p3 - p1
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def hull_area(points):
    """Calculate the area of the convex hull of the points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume # In 2D, volume is area
    except Exception:
        return 0.0

def score(points):
    """Calculate the normalized score: min triangle area / hull area."""
    h_area = hull_area(points)
    if h_area < 1e-10:
        return 0.0
    return min_triangle_area(points) / h_area

def construct():
    # Initialize with regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    current_points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    current_score = score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    # Simulated Annealing parameters
    T = 0.01
    T_min = 1e-5
    cooling = 0.995
    steps = 2000
    step_size = 0.05
    
    rng = np.random.default_rng(42)
    
    for _ in range(steps):
        # Pick a random point to move
        idx = rng.integers(0, N)
        old_pos = current_points[idx].copy()
        
        # Generate a new position
        # Small Gaussian perturbation
        new_pos = old_pos + rng.normal(0, step_size, 2)
        
        # Optional: Keep points within the unit disk? 
        # The problem says "normalized to unit area", but the evaluator normalizes by hull area.
        # So we don't strictly need to be inside a unit disk, but keeping them bounded helps.
        # Let's project back to unit disk if it goes outside to prevent explosion, 
        # although for this metric, scaling doesn't matter, only relative positions.
        # Actually, since the score is scale-invariant (ratio of areas), we can just let them move.
        # But to avoid numerical issues, let's keep them in a reasonable range.
        
        current_points[idx] = new_pos
        
        new_score = score(current_points)
        
        # Acceptance criterion
        if new_score > current_score or rng.random() < np.exp((new_score - current_score) / T):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = current_points.copy()
        else:
            # Revert
            current_points[idx] = old_pos
            
        T *= cooling
        if T < T_min:
            break
            
    return best_points
# EVOLVE-BLOCK-END
