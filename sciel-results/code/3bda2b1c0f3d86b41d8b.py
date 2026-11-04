"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize min triangle area."""
import numpy as np
import time
import random

N = 11

def _min_triangle_area(points):
    """Compute minimum triangle area among all C(n,3) triples."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = points[k] - points[i]
                b = points[j] - points[i]
                area = 0.5 * abs(a[0]*b[1] - a[1]*b[0])
                if area < min_area:
                    min_area = area
    return min_area

def _project_to_triangle(p):
    """Project point to the equilateral triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2)."""
    x, y = p
    h = np.sqrt(3) / 2
    # Check if inside
    if y < 0:
        y = 0
    if y > h:
        y = h
    # Left edge: from (0,0) to (0.5, h): y = sqrt(3)*x => x = y/sqrt(3)
    # Right edge: from (1,0) to (0.5, h): y = -sqrt(3)*(x-1) => x = 1 - y/sqrt(3)
    x_min = y / np.sqrt(3)
    x_max = 1 - y / np.sqrt(3)
    if x < x_min:
        x = x_min
    elif x > x_max:
        x = x_max
    return np.array([x, y])

def _initial_configuration():
    """Create a structured initial configuration using a grid-like pattern."""
    h = np.sqrt(3) / 2
    # Use a combination of boundary and interior points
    # Start with a roughly uniform distribution
    points = []
    
    # Add points along edges and interior in a structured way
    # Row 0 (bottom): 4 points
    for i in range(4):
        x = (i + 0.5) / 4.0
        points.append([x, 0.05])
    
    # Row 1 (middle): 3 points
    for i in range(3):
        x = 0.15 + i * 0.3
        points.append([x, 0.35])
    
    # Row 2 (upper): 2 points
    points.append([0.3, 0.55])
    points.append([0.7, 0.55])
    
    # Row 3 (top): 2 points
    points.append([0.4, 0.7])
    points.append([0.6, 0.7])
    
    # Project all points to triangle
    points = [_project_to_triangle(p) for p in points]
    return np.array(points)

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    rng = random.Random(42)
    np_rng = np.random.RandomState(42)
    
    # Start with structured configuration
    best_points = _initial_configuration()
    best_score = _min_triangle_area(best_points)
    
    # Simulated annealing
    current_points = best_points.copy()
    current_score = best_score
    
    T = 0.01  # Initial temperature
    T_min = 1e-8
    cooling = 0.995
    max_steps = 500000
    
    step = 0
    while step < max_steps:
        if time.time() > deadline:
            break
        
        # Random perturbation
        idx = rng.randint(0, N-1)
        move_size = T * 0.5
        dx = rng.gauss(0, move_size)
        dy = rng.gauss(0, move_size)
        
        new_point = current_points[idx].copy()
        new_point[0] += dx
        new_point[1] += dy
        new_point = _project_to_triangle(new_point)
        
        if np.allclose(new_point, current_points[idx]):
            step += 1
            continue
        
        trial_points = current_points.copy()
        trial_points[idx] = new_point
        trial_score = _min_triangle_area(trial_points)
        
        delta = trial_score - current_score
        
        if delta > 0 or rng.random() < np.exp(delta / T):
            current_points = trial_points
            current_score = trial_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        
        T *= cooling
        if T < T_min:
            T = T_min
        
        step += 1
    
    # Local refinement with smaller steps
    if time.time() < deadline - 5:
        refine_T = 0.001
        for _ in range(100000):
            if time.time() > deadline:
                break
            idx = rng.randint(0, N-1)
            move_size = refine_T * 0.1
            dx = rng.gauss(0, move_size)
            dy = rng.gauss(0, move_size)
            
            new_point = best_points[idx].copy()
            new_point[0] += dx
            new_point[1] += dy
            new_point = _project_to_triangle(new_point)
            
            if np.allclose(new_point, best_points[idx]):
                continue
            
            trial_points = best_points.copy()
            trial_points[idx] = new_point
            trial_score = _min_triangle_area(trial_points)
            
            if trial_score > best_score:
                best_score = trial_score
                best_points = trial_points
    
    return best_points
