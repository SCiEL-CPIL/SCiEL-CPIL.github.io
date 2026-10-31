import numpy as np
from scipy.spatial import ConvexHull
import time

N = 13

def compute_min_triangle_area(points):
    n = points.shape[0]
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def compute_hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def score(points):
    hull_area = compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_tri_area = compute_min_triangle_area(points)
    return min_tri_area / hull_area

def construct():
    # Start with a regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    current_points = points.copy()
    current_score = score(current_points)
    
    # Simulated Annealing parameters
    start_temp = 0.05
    end_temp = 1e-5
    time_limit = 110 # seconds
    start_time = time.time()
    
    # We will run until time limit
    while time.time() - start_time < time_limit:
        # Dynamic temperature: decrease over time
        elapsed = time.time() - start_time
        # Assume we want to finish annealing by time_limit
        # Simple linear or exponential decay based on elapsed time
        # Let's use a fixed number of iterations per second? No, just decay temp based on elapsed time.
        # temp = start_temp * exp(-elapsed / tau)
        # Let's try a simple decay
        temp = start_temp * (1 - elapsed / time_limit) ** 2
        if temp < end_temp:
            temp = end_temp
            
        # Pick a random point to move
        idx = np.random.randint(N)
        
        # Store original position
        original_pos = current_points[idx].copy()
        
        # Perturb with step size proportional to temp? Or fixed?
        # Fixed step size is often better for shape optimization
        step_size = 0.05
        noise = np.random.normal(0, step_size, 2)
        current_points[idx] = original_pos + noise
        
        new_score = score(current_points)
        
        # Acceptance criterion
        if new_score > current_score:
            current_score = new_score
        else:
            # Metropolis criterion
            # Avoid overflow in exp
            delta = new_score - current_score
            if delta < -100 * temp:
                prob = 0.0
            else:
                prob = np.exp(delta / temp)
                
            if np.random.random() < prob:
                current_score = new_score
            else:
                # Reject
                current_points[idx] = original_pos
                
    return current_points
