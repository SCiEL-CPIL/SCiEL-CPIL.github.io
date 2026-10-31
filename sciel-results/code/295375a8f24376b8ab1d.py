import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 11
H = np.sqrt(3) / 2.0
TRIAN_AREA = H * 1.0 / 2.0 # Base 1, Height H

# Pre-compute indices for all triangles
# i < j < k
TRIAN_IDX = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            TRIAN_IDX.append((i, j, k))
TRIAN_IDX = np.array(TRIAN_IDX) # Shape (165, 3)

def min_area(points):
    """
    points: np.array of shape (N, 2)
    Returns the minimum area of any triangle formed by 3 points.
    """
    # Extract points for all triangles
    # p1, p2, p3 shape (165, 2)
    p1 = points[TRIAN_IDX[:, 0]]
    p2 = points[TRIAN_IDX[:, 1]]
    p3 = points[TRIAN_IDX[:, 2]]
    
    # Vectors
    v1 = p2 - p1
    v2 = p3 - p1
    
    # Determinant (2x2)
    dets = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    
    areas = 0.5 * np.abs(dets)
    
    return np.min(areas)

def project_to_triangle(p):
    """
    Project point p (x, y) to the triangle with vertices (0,0), (1,0), (0.5, H).
    """
    x, y = p
    
    # Check if inside
    # 1. y >= 0
    # 2. y <= H * x  (Left edge: y = sqrt(3)*x => H = sqrt(3)/2, so y <= 2*H*x? No.
    #    Slope of left edge (0,0) to (0.5, H) is H/0.5 = 2H = sqrt(3).
    #    So y <= sqrt(3) * x.
    # 3. y <= H * (1 - x) * 2? No.
    #    Slope of right edge (1,0) to (0.5, H) is H / -0.5 = -2H = -sqrt(3).
    #    Equation: y - 0 = -sqrt(3) (x - 1) => y = sqrt(3) (1 - x).
    
    if y < 0:
        y = 0.0
    elif y > np.sqrt(3) * x:
        # Project onto line y = sqrt(3) x
        # Normal vector n = (sqrt(3), -1) ? No.
        # Line: sqrt(3)x - y = 0.
        # Distance d = (sqrt(3)x - y) / 2.
        # Projection: p - d * n / |n|
        # n = (sqrt(3), -1). |n| = 2.
        # d_signed = (sqrt(3)x - y) / 2.
        # Since y > sqrt(3)x, d_signed is negative.
        # We want to move towards the line.
        # Vector from p to line is -d_signed * n / |n|?
        # Let's use standard projection.
        # p' = p - ((sqrt(3)x - y) / 2) * (sqrt(3), -1) / 2 ?
        # Wait, line ax+by+c=0. a=sqrt(3), b=-1, c=0.
        # d = (ax+by+c)/sqrt(a^2+b^2) = (sqrt(3)x - y)/2.
        # p_proj = p - d * (a, b) / sqrt(a^2+b^2)
        # p_proj = p - d * (sqrt(3), -1) / 2
        d = (np.sqrt(3) * x - y) / 2.0
        x = x - d * (np.sqrt(3) / 2.0)
        y = y - d * (-1 / 2.0)
    elif y > np.sqrt(3) * (1 - x):
        # Project onto line y = sqrt(3)(1-x) => sqrt(3)x + y - sqrt(3) = 0
        # a=sqrt(3), b=1, c=-sqrt(3)
        d = (np.sqrt(3) * x + y - np.sqrt(3)) / 2.0
        x = x - d * (np.sqrt(3) / 2.0)
        y = y - d * (1 / 2.0)
        
    return np.array([x, y])

def hex_init(n):
    """
    Generate an initial hexagonal grid of n points.
    """
    # For n=11, use 4 rows (1,2,3,4) = 10 points + 1 center
    pts = []
    # Row 3 (base, y=0): 4 points
    for i in range(4):
        pts.append([i / 3.0, 0.0])
    # Row 2 (y=H/3): 3 points
    for i in range(3):
        pts.append([i / 3.0 + 1 / 6.0, H / 3.0])
    # Row 1 (y=2H/3): 2 points
    for i in range(2):
        pts.append([i / 3.0 + 1 / 3.0, 2 * H / 3.0])
    # Row 0 (y=H): 1 point
    pts.append([0.5, H])
    
    # Add 11th point at centroid
    pts.append([0.5, H / 3.0])
    
    return np.array(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # 1. Initialization
    points = hex_init(N)
    
    # 2. Simulated Annealing
    # Objective: maximize min_area
    current_score = min_area(points)
    best_points = points.copy()
    best_score = current_score
    
    T = 0.05
    step_size = 0.05
    # Cooling schedule
    cool_rate = 0.999
    
    # Pre-allocate arrays for speed? Not necessary for 11 points.
    
    while time.time() < deadline:
        # Pick a random point
        i = np.random.randint(N)
        
        # Generate random move
        # Gaussian noise
        noise = np.random.normal(0, step_size, 2)
        new_p = points[i] + noise
        
        # Project to triangle
        new_p = project_to_triangle(new_p)
        
        # Update point
        old_p = points[i].copy()
        points[i] = new_p
        
        # Calculate new score
        new_score = min_area(points)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        else:
            points[i] = old_p
        
        # Cool down
        T *= cool_rate
        if T < 0.001:
            T = 0.001
            # If T is too low, maybe restart with a new T or break
            # For now, keep it low to polish
            # Let's reset T occasionally if no improvement?
            # Simple approach: just keep going.
            
    # 3. Polish with Nelder-Mead
    # If time remains
    if time.time() < deadline - 10.0:
        # Flatten best_points
        x0 = best_points.flatten()
        
        def objective(x):
            pts = x.reshape(-1, 2)
            # Project
            for i in range(N):
                pts[i] = project_to_triangle(pts[i])
            return -min_area(pts)
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            # Check if result is better
            res_pts = res.x.reshape(-1, 2)
            # Ensure they are projected
            for i in range(N):
                res_pts[i] = project_to_triangle(res_pts[i])
            
            res_score = min_area(res_pts)
            if res_score > best_score:
                best_points = res_pts
                best_score = res_score
        except Exception:
            pass
            
    return best_points

