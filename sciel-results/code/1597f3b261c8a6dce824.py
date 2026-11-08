"""
Heilbronn triangles: maximize min triangle area for 11 points in unit-area equilateral triangle.
Strategy: Simulated Annealing with multiple random restarts and a structured initial guess.
This is a structurally different approach from Differential Evolution.
"""

import numpy as np
import time
import itertools
import random

N = 11
H = np.sqrt(3) / 2.0
AREA_EQUI = np.sqrt(3) / 4.0

# Pre-compute the 165 index triples for triangles
combos = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            combos.append((i, j, k))
combos = np.array(combos)
idx_i = combos[:, 0]
idx_j = combos[:, 1]
idx_k = combos[:, 2]

def min_triangle_area(points):
    """
    Vectorized computation of the minimum triangle area.
    points: shape (N, 2)
    """
    pi = points[idx_i]
    pj = points[idx_j]
    pk = points[idx_k]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def is_inside_triangle(points):
    """
    Check if all points are inside the unit equilateral triangle.
    Vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    """
    y = points[:, 1]
    x = points[:, 0]
    
    # Constraints for the triangle:
    # 1. y >= 0
    # 2. y <= sqrt(3) * x  (Left edge: from (0,0) to (0.5, H). Slope H/0.5 = sqrt(3). Region is below/right)
    # 3. y <= -sqrt(3) * (x - 1) = sqrt(3) * (1 - x) (Right edge: from (1,0) to (0.5, H). Slope -sqrt(3). Region is below/left)
    
    cond1 = y >= 0
    cond2 = y <= np.sqrt(3) * x
    cond3 = y <= np.sqrt(3) * (1 - x)
    
    return np.all(cond1) and np.all(cond2) and np.all(cond3)

def project_to_triangle(points):
    """
    Project points slightly outside the triangle back inside.
    Simple method: if outside, move towards centroid until inside (binary search or linear).
    For speed, we can just clamp coordinates roughly, but projection to edges is safer.
    However, for SA, we can also just reject invalid moves.
    This function is provided for completeness but might not be used in the main loop if rejection is faster.
    """
    # Simple repair: if outside, scale towards centroid
    centroid = np.array([0.5, H / 3.0])
    valid = is_inside_triangle(points)
    if not valid:
        for i in range(N):
            if not is_inside_triangle(points[i:i+1]):
                # Move point towards centroid
                # Find the largest t such that p + t*(centroid - p) is inside
                # Since centroid is strictly inside, t=1 is always valid.
                # We can just set it to centroid or interpolate.
                # A simple heuristic: if outside, move 50% towards centroid
                points[i] = points[i] * 0.5 + centroid * 0.5
                # Check again, if still outside, set to centroid
                if not is_inside_triangle(points[i:i+1]):
                    points[i] = centroid
    return points

def generate_structured_initial():
    """
    Generate a structured initial configuration.
    Based on known near-optimal configurations or symmetric patterns.
    For N=11, a common pattern is 3 points on each side (corners shared) or a grid-like structure.
    Let's try a 3x3 grid plus center? No, 3x3 is 9.
    Maybe 1 in center, 6 around, 4 on corners?
    Let's use a deterministic symmetric starting point which often helps SA escape local minima better than pure random.
    
    A good symmetric start:
    Center: (0.5, H/3)
    Ring 1: 6 points around center
    Ring 2: 4 points near corners?
    
    Actually, for Heilbronn, the optimal configurations are often "jagged".
    Let's just use a jittered grid or a simple symmetric arrangement.
    
    Arrangement:
    3 points on bottom edge: (0.1, 0.1), (0.5, 0.1), (0.9, 0.1) - wait, y=0 is boundary.
    Let's place them slightly inside.
    
    Let's try a 4-3-3-1 pattern?
    Or simply:
    3 points near each vertex, 1 in center, 2 on mid-edges?
    
    Let's stick to a simple symmetric set of 11 points.
    """
    pts = np.zeros((N, 2))
    
    # 1 center
    pts[0] = [0.5, H/3]
    
    # 6 points in a hexagon around center
    r = 0.2
    for i in range(6):
        angle = i * np.pi / 3
        # Scale y to match triangle aspect ratio? No, just use cartesian.
        # But we need to stay inside.
        x = 0.5 + r * np.cos(angle)
        y = H/3 + r * np.sin(angle) * (1.0) # y-scale is 1 in cartesian, but triangle is "taller" relative to width?
        # Actually, let's just use a fixed set of known good-looking coordinates.
        
    # Let's use a simpler structured start:
    # 3 points on each of the 3 sides (excluding corners), 1 in center, 2 in middle?
    
    # Side 1 (bottom): y=0.05
    pts[1] = [0.2, 0.05]
    pts[2] = [0.5, 0.05]
    pts[3] = [0.8, 0.05]
    
    # Side 2 (left): x = y/sqrt(3) + epsilon
    # Line from (0,0) to (0.5, H).
    # Points along this edge, inset slightly.
    # Param t in [0,1]. P = (0.5*t, H*t)
    # Inset by moving towards centroid.
    t_vals = [0.3, 0.5, 0.7]
    for i, t in enumerate(t_vals):
        px = 0.5 * t
        py = H * t
        # Move towards centroid (0.5, H/3)
        cx, cy = 0.5, H/3
        pts[4+i] = [px + 0.1*(cx-px), py + 0.1*(cy-py)]
        
    # Side 3 (right): Line from (1,0) to (0.5, H).
    # P = (1 - 0.5*t, H*t)
    for i, t in enumerate(t_vals):
        px = 1 - 0.5 * t
        py = H * t
        cx, cy = 0.5, H/3
        pts[7+i] = [px + 0.1*(cx-px), py + 0.1*(cy-py)]
        
    # We have 1 (center) + 3 (bottom) + 3 (left) + 3 (right) = 10 points.
    # Need 1 more. Place one in the middle of the triangle, slightly off-center?
    # Or just add one more to the center cluster?
    pts[10] = [0.5, H/3 + 0.1]
    
    # Ensure validity
    if not is_inside_triangle(pts):
        pts = project_to_triangle(pts)
        
    return pts

def generate_random_points():
    """
    Generate random points inside the triangle using barycentric coordinates.
    """
    pts = np.zeros((N, 2))
    for i in range(N):
        while True:
            u = random.random()
            v = random.random()
            if u + v <= 1.0:
                # Barycentric coordinates (u, v, 1-u-v)
                # p = u*A + v*B + (1-u-v)*C
                # A=(0,0), B=(1,0), C=(0.5, H)
                # p = u*(0,0) + v*(1,0) + (1-u-v)*(0.5, H)
                # x = v + 0.5*(1-u-v) = v + 0.5 - 0.5u - 0.5v = 0.5 - 0.5u + 0.5v
                # y = (1-u-v)*H
                x = 0.5 - 0.5*u + 0.5*v
                y = (1 - u - v) * H
                pts[i] = [x, y]
                break
    return pts

def local_optimize(points, max_iter=500):
    """
    Simple hill climbing with random perturbations.
    """
    current = points.copy()
    current_score = min_triangle_area(current)
    
    for _ in range(max_iter):
        improved = False
        # Try moving each point
        for i in range(N):
            # Random direction and magnitude
            angle = random.random() * 2 * np.pi
            magnitude = random.random() * 0.05
            dx = magnitude * np.cos(angle)
            dy = magnitude * np.sin(angle)
            
            new_pts = current.copy()
            new_pts[i][0] += dx
            new_pts[i][1] += dy
            
            if is_inside_triangle(new_pts):
                new_score = min_triangle_area(new_pts)
                if new_score > current_score:
                    current = new_pts
                    current_score = new_score
                    improved = True
        if not improved:
            break
            
    return current, current_score

def simulated_annealing(points, time_limit=10.0):
    """
    Simulated Annealing.
    """
    start_time = time.time()
    current = points.copy()
    current_score = min_triangle_area(current)
    best = current.copy()
    best_score = current_score
    
    T = 0.05  # Initial temperature
    cooling_rate = 0.99
    
    while time.time() - start_time < time_limit:
        # Generate neighbor
        neighbor = current.copy()
        i = random.randint(0, N-1)
        angle = random.random() * 2 * np.pi
        magnitude = T * 2.0  # Scale step size with temperature
        dx = magnitude * np.cos(angle)
        dy = magnitude * np.sin(angle)
        
        neighbor[i][0] += dx
        neighbor[i][1] += dy
        
        if is_inside_triangle(neighbor):
            neighbor_score = min_triangle_area(neighbor)
            delta = neighbor_score - current_score
            
            if delta > 0 or random.random() < np.exp(delta / T):
                current = neighbor
                current_score = neighbor_score
                
                if current_score > best_score:
                    best = current.copy()
                    best_score = current_score
        else:
            # If invalid, just skip or penalize. Skipping is fine for SA.
            pass
            
        T *= cooling_rate
        if T < 0.001:
            break
            
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    rng = np.random.RandomState(42)
    random.seed(42)
    
    best_points = None
    best_score = -np.inf
    
    # Phase 1: Try structured initial guess
    p0 = generate_structured_initial()
    p0, s0 = local_optimize(p0, max_iter=200)
    if s0 > best_score:
        best_score = s0
        best_points = p0
        
    # Phase 2: Simulated Annealing from structured guess
    p1, s1 = simulated_annealing(p0, time_limit=20.0)
    if s1 > best_score:
        best_score = s1
        best_points = p1
        
    # Phase 3: Multiple random restarts with SA
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() > deadline - 20.0:
            break
            
        p_rand = generate_random_points()
        # Quick local optimization
        p_rand, s_rand = local_optimize(p_rand, max_iter=50)
        
        # SA from this random start
        p_sa, s_sa = simulated_annealing(p_rand, time_limit=10.0)
        
        if s_sa > best_score:
            best_score = s_sa
            best_points = p_sa
            
    # Final local optimization on the best found
    if best_points is not None:
        final_pts, final_score = local_optimize(best_points, max_iter=500)
        if final_score > best_score:
            best_score = final_score
            best_points = final_pts
            
    # Fallback
    if best_points is None:
        best_points = generate_random_points()
        
    # Ensure validity
    if not is_inside_triangle(best_points):
        best_points = project_to_triangle(best_points)
        
    return best_points
