import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11
DEADLINE = 150.0
START_TIME = time.time()

def bary_to_cart(u, v):
    # Equilateral triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2)
    # Barycentric (u, v, w) where u+v+w=1
    # Cart = u*A + v*B + w*C
    # A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    # x = 0*u + 1*v + 0.5*w = v + 0.5*(1-u-v) = 0.5 - 0.5u + 0.5v
    # y = 0*u + 0*v + sqrt(3)/2 * w = sqrt(3)/2 * (1-u-v)
    sqrt3 = np.sqrt(3)
    w = 1 - u - v
    x = v + 0.5 * w
    y = 0.5 * sqrt3 * w
    return x, y

def cart_to_bary(x, y):
    sqrt3 = np.sqrt(3)
    # y = sqrt3/2 * w => w = 2y/sqrt3
    w = 2 * y / sqrt3
    # x = 0.5 - 0.5u + 0.5v => 2x = 1 - u + v => u - v = 1 - 2x
    # u + v = 1 - w
    # 2u = (1-w) + (1-2x) => u = 1 - 0.5w - x
    # v = 1 - w - u
    u = 1 - 0.5 * w - x
    v = 1 - w - u
    return u, v, w

def is_inside_triangle(u, v, tol=1e-5):
    w = 1 - u - v
    return (u >= -tol) & (v >= -tol) & (w >= -tol)

def min_triangle_area(points):
    # points: (N, 2) array
    n = len(points)
    min_area = float('inf')
    # Vectorized calculation of areas
    # Area = 0.5 * |det(P_j - P_i, P_k - P_i)|
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs(np.cross(p2 - p1, p3 - p1))
                if area < min_area:
                    min_area = area
    return min_area

def objective(params):
    # params: 2N array of barycentric u, v for each point
    u = params[0:N]
    v = params[N:2*N]
    
    # Check bounds
    if np.any(u < -1e-5) or np.any(v < -1e-5) or np.any(u + v > 1 + 1e-5):
        return 1e6
    
    # Convert to Cartesian
    x, y = bary_to_cart(u, v)
    points = np.column_stack((x, y))
    
    area = min_triangle_area(points)
    return -area

def generate_random_bary(n_points, rng):
    # Generate random points in barycentric coordinates
    # Method: generate 3 random numbers, normalize
    u = rng.random(n_points)
    v = rng.random(n_points)
    w = rng.random(n_points)
    s = u + v + w
    u /= s
    v /= s
    # w is implicit
    return u, v

def optimize_from_start(start_u, start_v, max_iter=2000):
    start_params = np.concatenate([start_u, start_v])
    try:
        res = minimize(objective, start_params, method='Nelder-Mead', 
                       options={'maxiter': max_iter, 'xatol': 1e-5, 'fatol': 1e-6})
        if res.success:
            best_u = res.x[0:N]
            best_v = res.x[N:2*N]
            # Clip to valid range
            best_u = np.clip(best_u, 0, 1)
            best_v = np.clip(best_v, 0, 1)
            # Re-normalize if slightly out due to clip? 
            # Nelder-Mead might drift. Let's project back to the triangle.
            # Simple projection: if u+v > 1, scale down? 
            # Or just check validity.
            if np.all(best_u >= -1e-5) and np.all(best_v >= -1e-5) and np.all(best_u + best_v <= 1 + 1e-5):
                x, y = bary_to_cart(best_u, best_v)
                points = np.column_stack((x, y))
                val = min_triangle_area(points)
                return points, val
    except Exception:
        pass
    return None, 0

def construct():
    global START_TIME, DEADLINE
    rng = np.random.default_rng(42)
    
    best_points = None
    best_val = 0
    
    # Initial candidates
    # 1. Random
    u1, v1 = generate_random_bary(N, rng)
    # 2. Grid-like
    # Generate points on a 4x4 grid in barycentric
    # Rows i=0..3, j=0..3-i
    # We need 11 points.
    # 4x4 grid has 10 points. Add center.
    grid_u = []
    grid_v = []
    for i in range(4):
        for j in range(4 - i):
            # u corresponds to vertex A (0,0)? 
            # Let's map i to u, j to v
            # u = i/3, v = j/3
            # This covers the triangle.
            grid_u.append(i / 3.0)
            grid_v.append(j / 3.0)
    # Add center
    grid_u.append(1/3)
    grid_v.append(1/3)
    u2 = np.array(grid_u)
    v2 = np.array(grid_v)
    
    # 3. Another random
    u3, v3 = generate_random_bary(N, rng)
    
    candidates = [
        (u1, v1),
        (u2, v2),
        (u3, v3)
    ]
    
    # Add some more random starts
    for _ in range(5):
        u, v = generate_random_bary(N, rng)
        candidates.append((u, v))
        
    for u, v in candidates:
        if time.time() > DEADLINE - 10:
            break
        points, val = optimize_from_start(u, v)
        if points is not None and val > best_val:
            best_val = val
            best_points = points
            
    if best_points is None:
        # Fallback to random
        u, v = generate_random_bary(N, rng)
        x, y = bary_to_cart(u, v)
        best_points = np.column_stack((x, y))
        
    return best_points

# EVOLVE-BLOCK-END
