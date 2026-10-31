import numpy as np
from scipy.spatial import ConvexHull
import time
from itertools import combinations

N = 13

def _triangle_area(p1, p2, p3):
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_triangle_area(points):
    n = points.shape[0]
    min_area = float('inf')
    # Use vectorized computation for speed
    for i in range(n - 2):
        p1 = points[i]
        for j in range(i + 1, n - 1):
            p2 = points[j]
            # Compute areas for all k > j at once
            p3s = points[j+1:]
            v1 = p2 - p1
            v2 = p3s - p1
            cross = v1[0] * v2[:, 1] - v1[1] * v2[:, 0]
            areas = 0.5 * np.abs(cross)
            min_area = min(min_area, np.min(areas))
    return min_area

def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_points = None
    best_score = -1.0
    
    # Strategy: Start with regular polygon, then use local search with random restarts
    # The regular 13-gon is a good starting point due to symmetry
    
    def regular_polygon(n, radius=1.0):
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    
    # Initial best: regular polygon
    init_pts = regular_polygon(N)
    init_score = _score(init_pts)
    best_points = init_pts.copy()
    best_score = init_score
    
    # Local search: perturb points slightly and accept if score improves
    def local_search(points, max_iter=200, step_size=0.05):
        current = points.copy()
        current_score = _score(current)
        best_local = current.copy()
        best_local_score = current_score
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Try random perturbations
            improved = False
            for _ in range(5):  # 5 random directions per iteration
                if time.time() > deadline:
                    break
                
                # Random direction in 2D for one random point
                idx = np.random.randint(N)
                direction = np.random.randn(2)
                direction /= np.linalg.norm(direction)
                
                new_point = current[idx].copy()
                new_point += step_size * direction
                candidate = current.copy()
                candidate[idx] = new_point
                
                cand_score = _score(candidate)
                
                if cand_score > current_score:
                    current = candidate
                    current_score = cand_score
                    improved = True
                    
                    if current_score > best_local_score:
                        best_local = current.copy()
                        best_local_score = current_score
            
            if not improved:
                # Reduce step size
                step_size *= 0.8
                if step_size < 0.001:
                    break
        
        return best_local, best_local_score
    
    # Multiple random restarts
    np.random.seed(42)
    n_restarts = 0
    while time.time() < deadline:
        n_restarts += 1
        
        # Create a perturbed starting point
        if n_restarts == 1:
            start_pts = regular_polygon(N)
        else:
            # Start from best found so far with some noise
            start_pts = best_points + np.random.randn(N, 2) * 0.1
        
        # Run local search
        local_best, local_score = local_search(start_pts, max_iter=100, step_size=0.05)
        
        if local_score > best_score:
            best_score = local_score
            best_points = local_best.copy()
        
        # Check if we should stop
        if time.time() > deadline:
            break
    
    # Final check
    final_score = _score(best_points)
    if final_score > best_score:
        best_score = final_score
    
    return best_points
