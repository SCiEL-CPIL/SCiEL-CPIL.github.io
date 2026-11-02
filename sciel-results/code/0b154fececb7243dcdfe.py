import numpy as np
from scipy.optimize import minimize
import itertools
import time

N = 11

# Vertices of the equilateral triangle
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3)/2])
TRIA_AREA = np.abs(np.cross(B-A, C-A)) / 2.0

def bary_to_cartesian(params):
    """
    params: shape (N, 2) containing (u, v) barycentric coords for each point.
    w = 1 - u - v.
    Returns cartesian coordinates.
    """
    u = params[:, 0]
    v = params[:, 1]
    w = 1.0 - u - v
    # Check validity
    if np.any(u < -1e-9) or np.any(v < -1e-9) or np.any(w < -1e-9):
        return None
    x = u * A[0] + v * B[0] + w * C[0]
    y = u * A[1] + v * B[1] + w * C[1]
    return np.column_stack((x, y))

def min_triangle_area(points):
    """
    Calculate the minimum area of any triangle formed by 3 points.
    points: shape (N, 2)
    """
    n = len(points)
    min_area = float('inf')
    # Vectorized calculation of all triangle areas
    # Indices i, j, k with i < j < k
    # We can use itertools.combinations for clarity, but vectorized is faster if N is large.
    # For N=11, combinations is fine.
    for i, j, k in itertools.combinations(range(n), 3):
        p1 = points[i]
        p2 = points[j]
        p3 = points[k]
        # Area = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
        area = 0.5 * np.abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
        if area < min_area:
            min_area = area
    return min_area

def objective(params):
    """
    Objective function for minimization.
    Returns negative of min area.
    Adds penalty if points are outside the triangle.
    """
    # params is a flat array of length 2*N
    u = params[0:N]
    v = params[N:2*N]
    
    # Check bounds for barycentric coords
    # u >= 0, v >= 0, u+v <= 1
    if np.any(u < 0) or np.any(v < 0) or np.any(u + v > 1.0 + 1e-9):
        # Return a large positive value (since we are minimizing -area, large positive means bad)
        # Actually, if we minimize -area, we want area to be large.
        # Invalid points should have a very small area (or penalty).
        # Let's return a large penalty value so the optimizer avoids it.
        return 1e6
    
    points = np.column_stack((
        u * A[0] + v * B[0] + (1-u-v) * C[0],
        u * A[1] + v * B[1] + (1-u-v) * C[1]
    ))
    
    area = min_triangle_area(points)
    return -area

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_params = None
    best_score = -1e9
    
    # Initial guesses
    # 1. Random
    # 2. Structured: Grid in barycentric coords
    
    # Generate a few initial points
    inits = []
    
    # Random init
    rng = np.random.default_rng(42)
    for _ in range(20):
        # Sample u, v such that u+v <= 1
        # Method: sample u, v from [0,1], if u+v>1, reflect or resample
        # Or sample s, t from [0,1], u = s*(1-t), v = t*(1-s) ? No.
        # Simple: sample u, v uniformly in triangle.
        # u = r1, v = r2*(1-r1)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        params = np.concatenate([u, v])
        inits.append(params)
        
    # Structured init: Try to place points in a symmetric grid
    # For N=11, maybe a 3x3 grid plus 2 points?
    # Let's try a few symmetric configurations manually or via simple rules.
    # A 3x3 grid in barycentric (u,v) has 9 points.
    # u in {0, 0.5, 1}, v in {0, 0.5, 1} -> 9 points.
    # But (1,1) is invalid (u+v=2).
    # Valid grid points in barycentric for u,v:
    # u=0, v=0 (A)
    # u=0, v=0.5
    # u=0, v=1 (C)
    # u=0.5, v=0
    # u=0.5, v=0.5
    # u=0.5, v=1 (Invalid? 0.5+1=1.5 > 1)
    # u=1, v=0 (B)
    # So a 3x3 grid doesn't fit nicely.
    
    # Let's just rely on the random restarts and Nelder-Mead.
    # Nelder-Mead is local, so many restarts are needed.
    
    methods = ['Nelder-Mead']
    
    current_init_idx = 0
    while time.time() < deadline:
        if current_init_idx >= len(inits):
            # Generate more inits
            for _ in range(20):
                u = rng.random(N)
                v = rng.random(N) * (1 - u)
                params = np.concatenate([u, v])
                inits.append(params)
        
        x0 = inits[current_init_idx]
        current_init_idx += 1
        
        # Run optimization
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if res.success or res.fun < best_score:
                # Check if the result is valid
                u = res.x[0:N]
                v = res.x[N:2*N]
                if np.all(u >= -1e-9) and np.all(v >= -1e-9) and np.all(u+v <= 1+1e-9):
                    # Clip small negatives
                    u = np.clip(u, 0, 1)
                    v = np.clip(v, 0, 1)
                    # Ensure u+v <= 1 by scaling if necessary? 
                    # Nelder-Mead with penalty should keep it valid.
                    # Let's re-calculate score to be sure
                    score = -res.fun
                    if score > best_score:
                        best_score = score
                        best_params = res.x.copy()
        except Exception:
            pass

    if best_params is None:
        # Fallback to random
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_params = np.concatenate([u, v])
        
    u = best_params[0:N]
    v = best_params[N:2*N]
    
    # Final cleanup: ensure points are strictly inside/on boundary
    # If u+v > 1 due to numerical error, project back
    # Simple projection: scale u,v by (1/(u+v)) if u+v > 1
    for i in range(N):
        if u[i] + v[i] > 1.0:
            scale = 1.0 / (u[i] + v[i])
            u[i] *= scale
            v[i] *= scale
            
    points = np.column_stack((
        u * A[0] + v * B[0] + (1-u-v) * C[0],
        u * A[1] + v * B[1] + (1-u-v) * C[1]
    ))
    
    return points

