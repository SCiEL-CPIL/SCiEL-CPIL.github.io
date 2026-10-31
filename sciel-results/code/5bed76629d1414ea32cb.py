"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import itertools

N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import itertools
import math

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

def point_in_triangle(p, v0=V0, v1=V1, v2=V2, eps=1e-9):
    """Check if point p is inside or on the triangle v0-v1-v2."""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    
    d1 = sign(p, v0, v1)
    d2 = sign(p, v1, v2)
    d3 = sign(p, v2, v0)
    
    has_neg = (d1 < -eps) or (d2 < -eps) or (d3 < -eps)
    has_pos = (d1 > eps) or (d2 > eps) or (d3 > eps)
    
    return not (has_neg and has_pos)

def min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the given points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def penalty_if_outside(points):
    """Return a penalty if any point is outside the triangle."""
    penalty = 0.0
    for p in points:
        if not point_in_triangle(p):
            # Compute distance to nearest edge
            # Edge V0-V1: y=0, distance = -y if y<0
            d1 = max(0, -p[1])
            # Edge V1-V2: line from (1,0) to (0.5, sqrt(3)/2)
            # Equation: sqrt(3)*x + y = sqrt(3)
            # Distance = (sqrt(3)*x + y - sqrt(3)) / sqrt(3+1) = (sqrt(3)*x + y - sqrt(3))/2
            d2 = max(0, (np.sqrt(3) * p[0] + p[1] - np.sqrt(3)) / 2)
            # Edge V2-V0: line from (0.5, sqrt(3)/2) to (0,0)
            # Equation: -sqrt(3)*x + y = 0
            # Distance = (-sqrt(3)*x + y) / 2
            d3 = max(0, (-np.sqrt(3) * p[0] + p[1]) / 2)
            
            penalty += d1 + d2 + d3
    return penalty

def objective_flat(x):
    """Objective function for flat parameter vector. Minimizes negative min area + penalty."""
    points = x.reshape(-1, 2)
    if penalty_if_outside(points) > 1e-6:
        return 10.0 + penalty_if_outside(points)
    min_area = min_triangle_area(points)
    return -min_area

def objective_vec(x):
    """Same as objective_flat but handles reshaping."""
    return objective_flat(x)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_points = None
    best_score = -float('inf')
    
    # Initial best: use a reasonable heuristic starting point
    # Start with vertices and some interior points
    initial_points = np.array([
        V0, V1, V2,
        [0.5, 0.0],
        [0.5, np.sqrt(3)/4],
        [0.25, np.sqrt(3)/4],
        [0.75, np.sqrt(3)/4],
        [0.5, np.sqrt(3)/2 * 0.75],
        [0.25, np.sqrt(3)/2 * 0.75],
        [0.75, np.sqrt(3)/2 * 0.75],
        [0.5, np.sqrt(3)/2 * 0.5]
    ])
    
    # Ensure all points are strictly inside (nudge if on boundary)
    for i in range(len(initial_points)):
        if not point_in_triangle(initial_points[i], eps=-1e-6):
            # Nudge toward centroid
            centroid = (V0 + V1 + V2) / 3
            initial_points[i] = 0.9 * initial_points[i] + 0.1 * centroid
    
    best_points = initial_points.copy()
    best_score = min_triangle_area(best_points)
    
    # Try differential evolution for a few seconds
    if time.time() < deadline - 20:
        try:
            bounds = [(0, 1), (0, np.sqrt(3)/2)] * N
            # Use a smaller population for speed
            de_result = differential_evolution(
                objective_flat,
                bounds=bounds,
                maxiter=50,
                popsize=15,
                tol=1e-4,
                seed=42,
                polish=True,
                init='sobol'
            )
            
            if de_result.fun < 1.0:  # Valid solution
                de_points = de_result.x.reshape(-1, 2)
                if penalty_if_outside(de_points) < 1e-6:
                    de_score = min_triangle_area(de_points)
                    if de_score > best_score:
                        best_score = de_score
                        best_points = de_points.copy()
        except Exception:
            pass
    
    # Local refinement using L-BFGS-B
    if time.time() < deadline - 10:
        try:
            x0 = best_points.flatten()
            res = minimize(
                objective_flat,
                x0,
                method='L-BFGS-B',
                bounds=[(0, 1), (0, np.sqrt(3)/2)] * N,
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            
            if res.fun < 1.0:
                res_points = res.x.reshape(-1, 2)
                if penalty_if_outside(res_points) < 1e-6:
                    res_score = min_triangle_area(res_points)
                    if res_score > best_score:
                        best_score = res_score
                        best_points = res_points.copy()
        except Exception:
            pass
    
    # Try a few random restarts with local optimization if time remains
    while time.time() < deadline - 5:
        # Generate a random starting point
        rng = np.random.RandomState(int(time.time() * 1000) % (2**31))
        rand_points = np.zeros((N, 2))
        for i in range(N):
            u, v = rng.random(), rng.random()
            while u + v >= 1:
                u, v = rng.random(), rng.random()
            rand_points[i] = [u + 0.5 * v, (np.sqrt(3) / 2) * v]
        
        # Nudge toward interior
        centroid = (V0 + V1 + V2) / 3
        rand_points = 0.95 * rand_points + 0.05 * centroid
        
        if penalty_if_outside(rand_points) > 1e-6:
            continue
        
        try:
            x0 = rand_points.flatten()
            res = minimize(
                objective_flat,
                x0,
                method='L-BFGS-B',
                bounds=[(0, 1), (0, np.sqrt(3)/2)] * N,
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            
            if res.fun < 1.0:
                res_points = res.x.reshape(-1, 2)
                if penalty_if_outside(res_points) < 1e-6:
                    res_score = min_triangle_area(res_points)
                    if res_score > best_score:
                        best_score = res_score
                        best_points = res_points.copy()
        except Exception:
            pass
        
        # Check if we should stop
        if time.time() > deadline - 5:
            break
    
    # Final check: ensure best_points is valid
    if penalty_if_outside(best_points) > 1e-6:
        # Fall back to a simple valid configuration
        best_points = np.array([
            [0.1, 0.05], [0.9, 0.05], [0.5, 0.05],
            [0.2, 0.2], [0.8, 0.2], [0.5, 0.2],
            [0.3, 0.4], [0.7, 0.4], [0.5, 0.4],
            [0.4, 0.6], [0.6, 0.6]
        ])
        # Ensure inside
        centroid = (V0 + V1 + V2) / 2
        for i in range(len(best_points)):
            if not point_in_triangle(best_points[i]):
                best_points[i] = 0.9 * best_points[i] + 0.1 * centroid
    
    return best_points
# EVOLVE-BLOCK-END
