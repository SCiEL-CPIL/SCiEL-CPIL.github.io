"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations
import sys

def _triangle_area(p1, p2, p3):
    """Compute the absolute area of triangle formed by p1, p2, p3."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_area(points):
    """Compute the minimum area of any triangle formed by three points."""
    n = len(points)
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _triangle_area(points[i], points[j], points[k])
                if a < min_a:
                    min_a = a
    return min_a

def _in_triangle(p, eps=1e-9):
    """Check if point p is inside or on the boundary of the equilateral triangle."""
    # Triangle vertices: A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    x, y = p
    # Barycentric coordinates
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3) / 2])
    
    # Using edge functions
    # Edge AB: y >= 0
    if y < -eps:
        return False
    # Edge BC: line from (1,0) to (0.5, sqrt(3)/2)
    # Direction: (-0.5, sqrt(3)/2), Normal pointing inward: (sqrt(3)/2, 0.5)
    # Equation: (x - 1)*sqrt(3)/2 + y*0.5 <= 0  => sqrt(3)*x + y <= sqrt(3)
    if np.sqrt(3) * x + y > np.sqrt(3) + eps:
        return False
    # Edge CA: line from (0.5, sqrt(3)/2) to (0,0)
    # Direction: (-0.5, -sqrt(3)/2), Normal pointing inward: (sqrt(3)/2, -0.5)
    # Equation: x*sqrt(3)/2 - y*0.5 >= 0 => sqrt(3)*x - y >= 0
    if np.sqrt(3) * x - y < -eps:
        return False
    return True

def _project_to_triangle(p):
    """Project point p onto the equilateral triangle if outside."""
    x, y = p
    # Simple clipping logic based on the inequalities derived above
    # 1. y >= 0
    if y < 0:
        y = 0
    # 2. sqrt(3)*x + y <= sqrt(3)
    if np.sqrt(3) * x + y > np.sqrt(3):
        # Move along normal (sqrt(3), 1) backwards
        val = (np.sqrt(3) * x + y - np.sqrt(3)) / (3 + 1)
        x -= val * np.sqrt(3)
        y -= val
    # 3. sqrt(3)*x - y >= 0
    if np.sqrt(3) * x - y < 0:
        val = (y - np.sqrt(3) * x) / (3 + 1)
        x += val * np.sqrt(3)
        y += val
    return [x, y]

def _objective_neg_min_area(points_flat):
    """Objective function for minimization: negative of min area."""
    points = points_flat.reshape(-1, 2)
    # Penalty for points outside triangle
    penalty = 0.0
    for p in points:
        if not _in_triangle(p, eps=-1e-6): # Slightly stricter for penalty
            # Distance to triangle boundary approximation
            # Just use a large penalty if clearly outside
            penalty += 1e5
            
    if penalty > 0:
        return penalty - _min_area(points)
        
    return -_min_area(points)

def construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin from 150s limit
    
    best_points = None
    best_val = -float('inf')
    
    # Initial diverse seeds
    seeds = []
    
    # Seed 1: Random
    rng = np.random.RandomState(42)
    pts = []
    for _ in range(N):
        while True:
            u, v = rng.random(2)
            if u + v < 1:
                p = [u + 0.5 * v, np.sqrt(3) / 2 * v]
                if _in_triangle(p):
                    pts.append(p)
                    break
    seeds.append(np.array(pts))
    
    # Seed 2: Grid-like (Voronoi-ish center)
    # Place points roughly on a hexagonal grid pattern
    pts = []
    # Center
    pts.append([0.5, np.sqrt(3)/6])
    # 6 points around center
    r = 0.2
    for i in range(6):
        angle = i * np.pi / 3
        x = 0.5 + r * np.cos(angle)
        y = np.sqrt(3)/6 + r * np.sin(angle)
        if _in_triangle([x, y]):
            pts.append([x, y])
    # Add 4 more points near corners/edges
    extra_candidates = [
        [0.1, 0.05], [0.9, 0.05], [0.5, np.sqrt(3)/2 - 0.1],
        [0.3, 0.2], [0.7, 0.2], [0.5, 0.1]
    ]
    for c in extra_candidates:
        if len(pts) >= N: break
        if _in_triangle(c):
            # Check distance to existing
            if all(np.linalg.norm(np.array(c) - np.array(p)) > 0.05 for p in pts):
                pts.append(c)
    # Fill remaining randomly if needed
    while len(pts) < N:
        u, v = rng.random(2)
        if u + v < 1:
            p = [u + 0.5 * v, np.sqrt(3) / 2 * v]
            if _in_triangle(p):
                pts.append(p)
    seeds.append(np.array(pts[:N]))
    
    # Seed 3: Perturbed Grid
    pts = []
    for i in range(3):
        for j in range(3):
            if len(pts) < N:
                # Barycentric to Cartesian
                u = i / 2.0 * 0.8 + 0.1
                v = j / 2.0 * 0.8 + 0.1
                if u + v < 0.9:
                    p = [u + 0.5 * v, np.sqrt(3) / 2 * v]
                    pts.append(p)
    while len(pts) < N:
        u, v = rng.random(2)
        if u + v < 1:
            p = [u + 0.5 * v, np.sqrt(3) / 2 * v]
            if _in_triangle(p):
                pts.append(p)
    seeds.append(np.array(pts[:N]))
    
    for seed in seeds:
        if time.time() > deadline:
            break
            
        current_points = seed.copy()
        current_val = _min_area(current_points)
        
        if current_val > best_val:
            best_val = current_val
            best_points = current_points.copy()
            
        # Local Optimization using Nelder-Mead or L-BFGS-B with penalty
        # We will use a penalty method to handle constraints
        # Objective: maximize min_area -> minimize -min_area
        # Constraint: points inside triangle
        
        def obj(x):
            pts = x.reshape(-1, 2)
            # Soft penalty for being outside
            pen = 0.0
            for p in pts:
                if not _in_triangle(p, eps=-1e-8):
                    # Calculate distance to boundary roughly
                    # This is a heuristic penalty
                    pen += 1000.0 * max(0, -min(
                        p[1], # dist to y=0 approx
                        np.sqrt(3) - (np.sqrt(3)*p[0] + p[1]), # dist to BC approx
                        np.sqrt(3)*p[0] - p[1] # dist to CA approx
                    ))
            val = _min_area(pts)
            return pen - val
        
        # Try Nelder-Mead for non-smooth min
        try:
            res = minimize(
                obj, 
                current_points.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            
            if res.success or res.fun < obj(current_points.flatten()):
                new_points = res.x.reshape(-1, 2)
                # Ensure points are strictly inside for next steps
                for i in range(len(new_points)):
                    if not _in_triangle(new_points[i]):
                        new_points[i] = _project_to_triangle(new_points[i])
                
                new_val = _min_area(new_points)
                if new_val > current_val:
                    current_points = new_points
                    current_val = new_val
                    
                    if current_val > best_val:
                        best_val = current_val
                        best_points = current_points.copy()
        except Exception:
            pass
            
        # Second pass: Polish with tighter tolerance
        if time.time() < deadline - 5:
            try:
                res2 = minimize(
                    obj, 
                    current_points.flatten(), 
                    method='Nelder-Mead', 
                    options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                if res2.fun < obj(current_points.flatten()):
                    new_points = res2.x.reshape(-1, 2)
                    for i in range(len(new_points)):
                        if not _in_triangle(new_points[i]):
                            new_points[i] = _project_to_triangle(new_points[i])
                    
                    new_val = _min_area(new_points)
                    if new_val > current_val:
                        current_points = new_points
                        current_val = new_val
                        
                        if current_val > best_val:
                            best_val = current_val
                            best_points = current_points.copy()
            except Exception:
                pass

    if best_points is None:
        # Fallback
        best_points = seeds[0]
        
    return best_points

# EVOLVE-BLOCK-END
