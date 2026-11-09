import numpy as np
from scipy.optimize import minimize

N = 11
H = np.sqrt(3) / 2

def min_triangle_area(points):
    n = points.shape[0]
    if n < 3:
        return 0.0
    # Use itertools.combinations for clarity, or indices
    # Vectorized version
    i, j, k = np.indices((n, n, n))
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    
    v1 = p2 - p1
    v2 = p3 - p1
    
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    
    return np.min(areas)

def is_inside_triangle(points):
    # points shape (N, 2)
    x, y = points[:, 0], points[:, 1]
    # y >= 0
    cond1 = y >= -1e-9
    # x >= 0
    cond2 = x >= -1e-9
    # x <= 1
    cond3 = x <= 1 + 1e-9
    # y <= H
    cond4 = y <= H + 1e-9
    # Left edge: y <= sqrt(3) * x
    # Right edge: y <= -sqrt(3) * (x - 1)
    # Combined: y <= min(sqrt(3)*x, -sqrt(3)*(x-1))
    # Note: sqrt(3)*x is valid for x in [0, 0.5]
    # -sqrt(3)*(x-1) is valid for x in [0.5, 1]
    # We can just check both and take the min, but the lines cross at x=0.5, y=H/2? No, y=H.
    # At x=0.5, sqrt(3)*0.5 = H/2. Wait.
    # Left edge: (0,0) to (0.5, H). Slope = H/0.5 = sqrt(3). y = sqrt(3)x.
    # Right edge: (1,0) to (0.5, H). Slope = (H-0)/(0.5-1) = -2H = -sqrt(3). y = -sqrt(3)(x-1).
    # At x=0.5, y = sqrt(3)*0.5 = H/2? No, H = sqrt(3)/2. So y = H.
    # Yes, they meet at (0.5, H).
    
    upper_bound = np.minimum(np.sqrt(3) * x, -np.sqrt(3) * (x - 1))
    cond5 = y <= upper_bound + 1e-9
    
    return np.all(cond1 & cond2 & cond3 & cond4 & cond5)

def objective(params):
    points = params.reshape(-1, 2)
    if not is_inside_triangle(points):
        return 1.0
    
    area = min_triangle_area(points)
    return -area

def construct():
    best_params = None
    best_score = -float('inf')
    
    # Use multiple random restarts to avoid local minima
    # 10 restarts should be sufficient and fast
    for seed in range(10):
        rng = np.random.RandomState(seed)
        points = np.zeros((N, 2))
        for i in range(N):
            while True:
                u = rng.random()
                v = rng.random()
                if u + v < 1:
                    x = u + 0.5 * v
                    y = H * v
                    points[i] = [x, y]
                    break
        
        params = points.flatten()
        
        res = minimize(objective, params, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-7})
        
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
    
    points = best_params.reshape(-1, 2)
    return points
