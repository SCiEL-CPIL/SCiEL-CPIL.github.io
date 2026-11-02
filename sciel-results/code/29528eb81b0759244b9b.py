"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _triangle_area(p1, p2, p3):
    """Compute area of triangle formed by three points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _convex_hull_area(points):
    """Compute the area of the convex hull of a set of points."""
    n = len(points)
    if n < 3:
        return 0.0
    
    # Andrew's monotone chain algorithm
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    # Sort by x, then y
    sorted_pts = sorted(points, key=lambda p: (p[0], p[1]))
    
    lower = []
    for p in sorted_pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    upper = []
    for p in reversed(sorted_pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    hull = lower[:-1] + upper[:-1]
    
    if len(hull) < 3:
        return 0.0
    
    # Shoelace formula
    area = 0.0
    m = len(hull)
    for i in range(m):
        j = (i + 1) % m
        area += hull[i][0] * hull[j][1]
        area -= hull[j][0] * hull[i][1]
    return abs(area) / 2.0

def _min_triangle_area(points):
    """Compute the minimum triangle area among all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            for k in range(j + 1, n):
                pk = points[k]
                area = _triangle_area(pi, pj, pk)
                if area < min_area:
                    min_area = area
    return min_area

def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area

def construct():
    """
    Place 14 points in a planar convex region to maximize the minimum triangle area
    normalized by the convex hull area.
    
    Strategy: Simulated Annealing
    """
    start_time = time.time()
    max_time = 100.0 # Leave 20s buffer
    
    # Initial configuration: Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    current_points = np.column_stack([np.cos(angles), np.sin(angles)])
    current_score = _score(current_points)
    
    best_points = current_points.copy()
    best_score = current_score
    
    # Simulated Annealing parameters
    T = 1.0 # Initial temperature
    T_min = 1e-6
    alpha = 0.995 # Cooling rate
    max_iter = 50000
    
    rng = random.Random(42)
    
    for i in range(max_iter):
        if time.time() - start_time > max_time:
            break
            
        # Generate neighbor
        # Perturb all points slightly
        # Step size depends on temperature
        step_size = 0.1 * T + 0.01
        new_points = current_points + np.random.randn(N, 2) * step_size
        
        # Optional: Normalize scale to keep points in a reasonable range?
        # The problem is scale invariant (ratio), so we don't strictly need to constrain bounds,
        # but keeping them centered helps numerical stability.
        # Center the points
        new_points -= np.mean(new_points, axis=0)
        
        new_score = _score(new_points)
        
        # Acceptance criterion
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        else:
            # Metropolis criterion: accept with probability exp((new - current) / T)
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    current_points = new_points
                    current_score = new_score
            else:
                # Should not happen if new_score <= current_score, but just in case
                current_points = new_points
                current_score = new_score
        
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
    
    return best_points
# EVOLVE-BLOCK-END
