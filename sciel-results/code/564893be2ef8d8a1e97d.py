import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc

N = 11
SQRT3 = np.sqrt(3)

def _square_to_triangle(s):
    """
    Bijective mapping from (0,1)^2 to equilateral triangle.
    s: (N, 2)
    Returns: (N, 2) Cartesian coordinates
    """
    x = s[:, 0]
    y = s[:, 1]
    # Standard mapping:
    # x_tri = x + 0.5 * y
    # y_tri = (sqrt(3)/2) * y
    # But we need to ensure we cover the triangle (0,0), (1,0), (0.5, sqrt(3)/2)
    # Let's use the standard isometry from unit square to unit triangle.
    # Actually, a simpler mapping is:
    # u = x, v = y
    # Point = u * (1,0) + v * (0.5, sqrt(3)/2) + (1-u-v) * (0,0)
    # This requires u+v <= 1.
    # The map (x,y) -> (x, y*(1-x)) maps (0,1)^2 to the triangle.
    # Let's verify:
    # If x=0, y=0 -> (0,0)
    # If x=1, y=0 -> (1,0)
    # If x=0, y=1 -> (0, 1) -> Wait, vertex is (0.5, sqrt(3)/2).
    
    # Correct mapping from (0,1)^2 to triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2):
    # Let p1 = (0,0), p2 = (1,0), p3 = (0.5, sqrt(3)/2).
    # Any point P = u*p1 + v*p2 + w*p3 with u+v+w=1, u,v,w>=0.
    # We can parameterize u,v,w by two variables.
    # Let x in (0,1), y in (0,1).
    # w = y * (1-x) ? No.
    
    # Standard bijection:
    # x_tri = x + 0.5*y
    # y_tri = (sqrt(3)/2)*y
    # This maps (0,1)^2 to a parallelogram, not a triangle.
    
    # Let's use the method:
    # u = 1 - x
    # v = x * (1 - y)
    # w = x * y
    # Sum = 1.
    # P = u*(0,0) + v*(1,0) + w*(0.5, sqrt(3)/2)
    # P_x = v * 1 + w * 0.5 = x(1-y) + 0.5*xy = x - xy + 0.5xy = x - 0.5xy
    # P_y = w * sqrt(3)/2 = 0.5*sqrt(3)*xy
    
    # Check vertices:
    # (0,0) -> x=0,y=0 -> (0,0). OK.
    # (1,0) -> x=1,y=0 -> (1,0). OK.
    # (0.5, sqrt(3)/2) -> x=1,y=1 -> (1 - 0.5, 0.5*sqrt(3)) = (0.5, sqrt(3)/2). OK.
    
    px = s[:, 0] - 0.5 * s[:, 0] * s[:, 1]
    py = 0.5 * SQRT3 * s[:, 0] * s[:, 1]
    return np.column_stack([px, py])

def _triangle_to_square(p):
    """
    Inverse mapping from triangle to (0,1)^2.
    p: (N, 2)
    """
    px = p[:, 0]
    py = p[:, 1]
    
    # From py = 0.5 * SQRT3 * x * y
    # x * y = py / (0.5 * SQRT3) = 2 * py / SQRT3
    # From px = x - 0.5 * x * y
    # px = x - 0.5 * (2 * py / SQRT3) = x - py / SQRT3
    # x = px + py / SQRT3
    
    # Then y = (2 * py / SQRT3) / x
    
    x = px + py / SQRT3
    # Handle x=0 case (shouldn't happen for interior points, but safe)
    x_safe = np.where(x < 1e-12, 1e-12, x)
    y = (2 * py / SQRT3) / x_safe
    
    # Clip to [0,1] to handle numerical errors
    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, 1)
    
    return np.column_stack([x, y])

def _min_triangle_area(points):
    """
    Calculate the minimum area of any triangle formed by 3 points.
    points: (N, 2)
    """
    min_area = float('inf')
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * |det(p2-p1, p3-p1)|
                v1 = p2 - p1
                v2 = p3 - p1
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _objective(s):
    """
    Objective function for minimization (we want to maximize min area, so minimize negative).
    s: (N, 2) in (0,1)^2
    """
    points = _square_to_triangle(s)
    min_area = _min_triangle_area(points)
    return -min_area

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Use Sobol sequence for global exploration
    sampler = qmc.Sobol(d=2 * N, scramble=True, seed=42)
    
    # Number of samples for global search
    # We can't generate too many because optimization is expensive.
    # Let's try 50-100 samples.
    num_samples = 100
    
    samples = sampler.random(n=num_samples)
    
    best_val = -1e9
    best_s = None
    
    # We will iterate through samples, but we need to be careful with time.
    # Nelder-Mead can be slow. Let's limit maxiter.
    
    # To save time, we might not be able to run full NM on all 100.
    # Let's run NM on a subset, or use a simpler local search first?
    # Nelder-Mead is derivative-free and robust.
    
    # Strategy:
    # 1. Evaluate all samples to find top K candidates.
    # 2. Run Nelder-Mead on top K candidates.
    
    vals = np.array([_objective(s) for s in samples])
    
    # Sort by value (most negative is best)
    sorted_indices = np.argsort(vals)
    
    # Take top 5 candidates for refinement
    top_k = 5
    for idx in sorted_indices[:top_k]:
        if time.time() > deadline - 10:
            break
        
        s0 = samples[idx]
        # Run Nelder-Mead
        try:
            res = minimize(_objective, s0, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.success:
                val = res.fun
                if val < best_val:
                    best_val = val
                    best_s = res.x
        except Exception:
            pass
            
    # If no success, fallback to best sample
    if best_s is None:
        best_s = samples[sorted_indices[0]]
        
    # Convert best_s to Cartesian
    points = _square_to_triangle(best_s)
    
    return points

