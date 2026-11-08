import numpy as np
import time
from scipy.spatial import ConvexHull
import random

N = 13

def compute_min_triangle_area(points):
    """
    Compute the minimum area of all triangles formed by the points.
    points: (N, 2) array
    Returns: min_area
    """
    n = len(points)
    # Get indices for all unique pairs (i, j) with i < j < k
    # It's easier to just loop or use meshgrid
    # For N=13, 286 triangles is small.
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * |det(p2-p1, p3-p1)|
                dx1 = p2[0] - p1[0]
                dy1 = p2[1] - p1[1]
                dx2 = p3[0] - p1[0]
                dy2 = p3[1] - p1[1]
                area = 0.5 * abs(dx1 * dy2 - dy1 * dx2)
                areas.append(area)
    return min(areas)

def compute_hull_area(points):
    """
    Compute the area of the convex hull of the points.
    points: (N, 2) array
    Returns: hull_area
    """
    try:
        hull = ConvexHull(points)
        return hull.volume # In 2D, volume is area
    except Exception:
        # If points are collinear or something, return 0 or small epsilon
        return 1e-10

def normalize_points(points, target_area=1.0):
    """
    Scale points such that the convex hull area is target_area.
    """
    current_area = compute_hull_area(points)
    if current_area < 1e-10:
        return points
    scale = np.sqrt(target_area / current_area)
    return points * scale

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Leave 5s margin for cleanup/return
    
    # 1. Initialization: Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    # Start with radius 1
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Normalize to unit hull area
    points = normalize_points(points, 1.0)
    
    # Calculate initial score
    min_area = compute_min_triangle_area(points)
    hull_area = compute_hull_area(points)
    best_score = min_area / hull_area
    best_points = points.copy()
    
    # 2. Optimization Loop
    # Simulated Annealing / Hill Climbing
    # We will use a simple hill climbing with occasional random jumps (restarts) if stuck
    current_points = best_points.copy()
    current_min_area = min_area
    current_hull_area = hull_area
    current_score = best_score
    
    # Temperature for Simulated Annealing
    T = 0.01
    T_min = 1e-6
    cooling_rate = 0.999
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        # Pick a random point to perturb
        idx = rng.integers(0, N)
        
        # Perturbation magnitude
        # Start with small steps, maybe decrease over time?
        # Or keep constant small steps.
        step_size = 0.01
        # Add some randomness to step size
        step_size *= rng.uniform(0.5, 1.5)
        
        # Random direction
        angle = rng.uniform(0, 2 * np.pi)
        dx = step_size * np.cos(angle)
        dy = step_size * np.sin(angle)
        
        new_points = current_points.copy()
        new_points[idx][0] += dx
        new_points[idx][1] += dy
        
        # Calculate new metrics
        # Optimization: We can check if the new point creates a triangle with area 
        # smaller than current_min_area * (current_hull_area / new_hull_area) ?
        # No, just compute fully. It's fast enough.
        
        new_hull_area = compute_hull_area(new_points)
        if new_hull_area < 1e-10:
            continue # Invalid move
            
        new_min_area = compute_min_triangle_area(new_points)
        new_score = new_min_area / new_hull_area
        
        # Acceptance criteria
        if new_score > current_score:
            current_points = new_points
            current_min_area = new_min_area
            current_hull_area = new_hull_area
            current_score = new_score
            
            # Update best
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
            
            # Normalize to keep scale stable? 
            # Since score is scale invariant, we don't strictly need to, 
            # but it helps prevent drift if we were doing gradient descent.
            # For random walk, scale drift is slow. 
            # However, if we normalize, we change ALL points. 
            # Let's NOT normalize inside the loop to save time, 
            # as the ratio is invariant. The absolute coordinates might drift, 
            # but N=13 is small.
        else:
            # Simulated Annealing acceptance
            delta = current_score - new_score
            if delta < 0:
                pass # Should not happen if we checked >
            else:
                # Accept with probability exp(-delta / T)
                if rng.random() < np.exp(-delta / T):
                    current_points = new_points
                    current_min_area = new_min_area
                    current_hull_area = new_hull_area
                    current_score = new_score
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        # Optional: Random restarts if stuck? 
        # If no improvement for a long time, reset to best and start again?
        # Given time constraints, simple hill climbing/SA is robust.
        
    # Final normalization to ensure hull area is exactly 1.0 (though score is invariant, 
    # the task says "normalized to unit area", so returning points with hull area 1 is good practice)
    best_points = normalize_points(best_points, 1.0)
    
    return best_points

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import random

N = 13

def _compute_min_triangle_area(points):
    n = len(points)
    min_area = float('inf')
    # Vectorized approach for speed
    # Create indices
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    # Filter i < j < k
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_f = i_idx[mask]
    j_f = j_idx[mask]
    k_f = k_idx[mask]
    
    p1 = points[i_f]
    p2 = points[j_f]
    p3 = points[k_f]
    
    # Cross product in 2D
    # (p2 - p1) x (p3 - p1)
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _compute_hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def _normalize_points(points, target_area=1.0):
    current_area = _compute_hull_area(points)
    if current_area < 1e-10:
        return points
    scale = np.sqrt(target_area / current_area)
    return points * scale

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # 1. Initialization: Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    points = _normalize_points(points, 1.0)
    
    current_min_area = _compute_min_triangle_area(points)
    current_hull_area = _compute_hull_area(points)
    current_score = current_min_area / current_hull_area
    
    best_score = current_score
    best_points = points.copy()
    current_points = points.copy()
    
    rng = np.random.default_rng(42)
    
    # Parameters
    T = 0.005
    T_min = 1e-7
    cooling_rate = 0.9995
    
    while time.time() < deadline:
        idx = rng.integers(0, N)
        
        # Adaptive step size? Start small.
        step_size = 0.01 * rng.uniform(0.5, 2.0)
        angle = rng.uniform(0, 2 * np.pi)
        dx = step_size * np.cos(angle)
        dy = step_size * np.sin(angle)
        
        new_points = current_points.copy()
        new_points[idx][0] += dx
        new_points[idx][1] += dy
        
        new_hull_area = _compute_hull_area(new_points)
        if new_hull_area < 1e-10:
            continue
            
        new_min_area = _compute_min_triangle_area(new_points)
        new_score = new_min_area / new_hull_area
        
        if new_score > current_score:
            current_points = new_points
            current_min_area = new_min_area
            current_hull_area = new_hull_area
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        else:
            delta = current_score - new_score
            if delta > 0:
                if rng.random() < np.exp(-delta / T):
                    current_points = new_points
                    current_min_area = new_min_area
                    current_hull_area = new_hull_area
                    current_score = new_score
        
        T *= cooling_rate
        if T < T_min:
            T = T_min

    best_points = _normalize_points(best_points, 1.0)
    return best_points
# EVOLVE-BLOCK-END
