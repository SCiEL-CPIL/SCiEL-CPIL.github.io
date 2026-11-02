"""Heilbronn triangles: place 11 points in a unit equilateral triangle to maximize the minimum triangle area."""
import numpy as np
import math
import time

N = 11
S3 = math.sqrt(3)

def _project_to_triangle(points):
    """
    Project points onto the equilateral triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2).
    """
    points = points.copy()
    # y >= 0
    points[points[:, 1] < 0, 1] = 0
    
    # y <= sqrt(3) * x  =>  sqrt(3)*x - y >= 0
    # If violated, project onto y = sqrt(3)*x
    # The line is sqrt(3)*x - y = 0. Normal is (sqrt(3), -1).
    # Distance projection: p - d * n / |n|^2
    # d = sqrt(3)*x - y
    # n = (sqrt(3), -1), |n|^2 = 3+1=4
    # x_new = x - d * sqrt(3) / 4
    # y_new = y - d * (-1) / 4 = y + d/4
    # Check: y_new = y + (sqrt(3)x - y)/4 = 3y/4 + sqrt(3)x/4. 
    # Is y_new = sqrt(3) x_new?
    # sqrt(3) (x - d sqrt(3)/4) = sqrt(3)x - 3d/4 = sqrt(3)x - 3(sqrt(3)x - y)/4 = sqrt(3)x/4 + 3y/4. Yes.
    
    d1 = S3 * points[:, 0] - points[:, 1]
    mask1 = d1 < 0
    if np.any(mask1):
        d1_masked = d1[mask1]
        points[mask1, 0] -= d1_masked * S3 / 4.0
        points[mask1, 1] += d1_masked / 4.0
        
    # y <= sqrt(3) * (1 - x) => sqrt(3) - sqrt(3)*x - y >= 0
    # Line: -sqrt(3)*x - y + sqrt(3) = 0. Normal (-sqrt(3), -1).
    # d = -sqrt(3)*x - y + sqrt(3)
    # x_new = x - d * (-sqrt(3)) / 4 = x + d * sqrt(3) / 4
    # y_new = y - d * (-1) / 4 = y + d / 4
    
    d2 = -S3 * points[:, 0] - points[:, 1] + S3
    mask2 = d2 < 0
    if np.any(mask2):
        d2_masked = d2[mask2]
        points[mask2, 0] += d2_masked * S3 / 4.0
        points[mask2, 1] += d2_masked / 4.0
        
    return points

def _min_area(points):
    """
    Compute minimum area of any triangle formed by 3 points.
    """
    n = len(points)
    min_area = float('inf')
    # Precompute differences? 
    # Loop over i, j, k
    for i in range(n):
        p1 = points[i]
        for j in range(i + 1, n):
            p2 = points[j]
            v1 = p2 - p1
            for k in range(j + 1, n):
                p3 = points[k]
                v2 = p3 - p1
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def construct():
    """
    Place 11 points using Simulated Annealing.
    """
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    best_points = None
    best_score = -float('inf')
    
    # Initial configuration: Centroid + small noise
    # Or a known good configuration?
    # Let's use random restarts.
    
    n_restarts = 50
    np.random.seed(42)
    
    for r in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize
        if r == 0:
            # Start with a regular grid or centroid cluster
            points = np.full((N, 2), [0.5, S3/6])
            # Add small noise to break symmetry
            points += np.random.normal(0, 0.05, (N, 2))
        else:
            points = np.zeros((N, 2))
            for i in range(N):
                x = np.random.uniform(0, 1)
                y_max = S3 * min(x, 1 - x)
                y = np.random.uniform(0, y_max)
                points[i] = [x, y]
        
        points = _project_to_triangle(points)
        
        current_area = _min_area(points)
        current_energy = -current_area
        best_local_area = current_area
        best_local_points = points.copy()
        
        # Simulated Annealing parameters
        T = 0.05
        T_min = 1e-6
        alpha = 0.995
        step_size = 0.05
        
        while T > T_min and (time.time() - start_time) < time_limit:
            # Generate a move: perturb a random point
            idx = np.random.randint(N)
            new_points = points.copy()
            
            # Random direction
            dx = np.random.normal(0, step_size)
            dy = np.random.normal(0, step_size)
            
            new_points[idx, 0] += dx
            new_points[idx, 1] += dy
            
            new_points = _project_to_triangle(new_points)
            
            new_area = _min_area(new_points)
            new_energy = -new_area
            
            delta_energy = new_energy - current_energy
            
            # Metropolis acceptance
            if delta_energy < 0 or (np.random.random() < math.exp(-delta_energy / T)):
                points = new_points
                current_energy = new_energy
                current_area = new_area
                
                if current_area > best_local_area:
                    best_local_area = current_area
                    best_local_points = points.copy()
            
            T *= alpha
            
        # Update global best
        if best_local_area > best_score:
            best_score = best_local_area
            best_points = best_local_points
    
    if best_points is None:
        return np.full((N, 2), [0.5, S3/6])
        
    return best_points
