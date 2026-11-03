"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations

# Triangle vertices
V1 = np.array([0.0, 0.0])
V2 = np.array([1.0, 0.0])
V3 = np.array([0.5, np.sqrt(3)/2])

def point_in_triangle(p, a=V1, b=V2, c=V3):
    """Check if point p is inside or on the triangle abc."""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)

def triangle_area(p1, p2, p3):
    """Area of triangle formed by three points."""
    return 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))

def min_triangle_area(points):
    """Minimum area of any triangle formed by three points."""
    n = len(points)
    min_area = float('inf')
    for i, j, k in combinations(range(n), 3):
        area = triangle_area(points[i], points[j], points[k])
        if area < min_area:
            min_area = area
    return min_area

def initial_configuration():
    """Create a structured initial configuration for 11 points in the triangle."""
    # Use a grid-like arrangement: rows with decreasing number of points
    # Row 0 (bottom): 4 points, Row 1: 4 points, Row 2: 3 points
    points = []
    # Bottom row: y = small epsilon, x from 0.1 to 0.9
    y0 = 0.05
    for i in range(4):
        x = 0.1 + i * 0.2
        points.append(np.array([x, y0]))
    # Middle row: y = sqrt(3)/4 ~ 0.433, x from 0.2 to 0.8
    y1 = np.sqrt(3)/4
    for i in range(4):
        x = 0.2 + i * 0.2
        points.append(np.array([x, y1]))
    # Top row: y = sqrt(3)/2 * 0.8 ~ 0.693, x around 0.5
    y2 = np.sqrt(3)/2 * 0.8
    for i in range(3):
        x = 0.35 + i * 0.15
        points.append(np.array([x, y2]))
    
    # Clip points to be inside the triangle
    clipped = []
    for p in points:
        if not point_in_triangle(p):
            # Project to nearest edge or move inward
            # Simple: scale toward centroid
            centroid = (V1 + V2 + V3) / 3
            t = 0.9
            p_new = centroid + t * (p - centroid)
            clipped.append(p_new)
        else:
            clipped.append(p)
    return np.array(clipped)

def construct():
    np.random.seed(42)
    points = initial_configuration()
    
    # Simulated annealing
    current = points.copy()
    current_score = min_triangle_area(current)
    best = current.copy()
    best_score = current_score
    
    T = 0.1  # Initial temperature
    T_min = 1e-6
    cooling = 0.995
    max_iter = 50000
    step_size = 0.05
    
    for it in range(max_iter):
        if T < T_min:
            break
        
        # Pick a random point to move
        idx = np.random.randint(0, N)
        old_point = current[idx].copy()
        
        # Random perturbation
        delta = np.random.normal(0, step_size, 2)
        new_point = old_point + delta
        
        # Check if new point is inside triangle
        if point_in_triangle(new_point):
            current[idx] = new_point
            new_score = min_triangle_area(current)
            
            # Accept if better or with probability exp((new-old)/T)
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best = current.copy()
            else:
                prob = np.exp((new_score - current_score) / T)
                if np.random.random() < prob:
                    current_score = new_score
                else:
                    current[idx] = old_point
        
        T *= cooling
    
    return best
# EVOLVE-BLOCK-END
