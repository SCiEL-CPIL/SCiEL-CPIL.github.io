import numpy as np
import random
from scipy.spatial import ConvexHull

N = 13

def construct():
    # Initial guess: Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Precompute indices for triangles
    # We need all combinations (i, j, k) with i < j < k
    tri_indices = []
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                tri_indices.append((i, j, k))
    tri_indices = np.array(tri_indices)
    tri_i = tri_indices[:, 0]
    tri_j = tri_indices[:, 1]
    tri_k = tri_indices[:, 2]
    
    def score(pts):
        # Vectorized triangle area calculation
        p_i = pts[tri_i]
        p_j = pts[tri_j]
        p_k = pts[tri_k]
        
        v1 = p_j - p_i
        v2 = p_k - p_i
        # Cross product in 2D: x1*y2 - y1*x2
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        min_area = np.min(areas)
        
        if min_area < 1e-12:
            return 0.0
            
        # Convex Hull Area
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume
        except Exception:
            # Fallback if points are collinear or degenerate
            return 0.0
            
        return min_area / hull_area

    best_points = points.copy()
    best_score = score(points)
    
    # Simulated Annealing Parameters
    T_start = 0.1
    T_end = 0.0001
    cooling_rate = 0.998
    steps = 10000
    
    rng = np.random.default_rng(42)
    
    current_points = points.copy()
    current_score = best_score
    T = T_start
    
    for step in range(steps):
        # Pick a random point to move
        idx = rng.integers(0, N)
        
        # Save original position
        orig_point = current_points[idx].copy()
        
        # Perturb the point
        # Adaptive step size
        step_size = 0.05 * (T / T_start) + 0.001
        current_points[idx] = orig_point + rng.normal(0, step_size, 2)
        
        new_score = score(current_points)
        
        # Acceptance criterion
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = current_points.copy()
        else:
            # Metropolis criterion
            delta = current_score - new_score
            if delta < 0:
                accept = True
            else:
                # Avoid division by zero or overflow
                if T > 0:
                    prob = np.exp(-delta / T)
                else:
                    prob = 0.0
                accept = (rng.random() < prob)
            
            if not accept:
                # Revert
                current_points[idx] = orig_point
        
        T *= cooling_rate
        
        if T < T_end:
            break
            
    return best_points
