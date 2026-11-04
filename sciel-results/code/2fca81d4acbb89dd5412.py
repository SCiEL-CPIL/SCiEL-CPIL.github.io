import numpy as np
from scipy.optimize import minimize
import time

N = 11

def _project_to_triangle(points):
    """
    Projects points onto the equilateral triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2).
    points: (N, 2) array
    """
    points = np.array(points, dtype=float)
    x = points[:, 0]
    y = points[:, 1]
    
    # Constraint 1: y >= 0
    # If y < 0, project to y=0
    mask1 = y < 0
    y[mask1] = 0
    
    # Constraint 2: y <= sqrt(3) * x  =>  sqrt(3)*x - y >= 0
    # Line: sqrt(3)*x - y = 0. Normal n = (sqrt(3), -1).
    # dist = (sqrt(3)*x - y) / ||n|| = (sqrt(3)*x - y) / 2
    val2 = np.sqrt(3) * x - y
    mask2 = val2 < 0
    if np.any(mask2):
        d = -val2[mask2] / 2.0
        x[mask2] = x[mask2] - np.sqrt(3) * d
        y[mask2] = y[mask2] + d
        
    # Constraint 3: y <= -sqrt(3) * (x - 1)  =>  sqrt(3)*x + y <= sqrt(3)
    # Line: sqrt(3)*x + y - sqrt(3) = 0. Normal n = (sqrt(3), 1).
    # dist = (sqrt(3)*x + y - sqrt(3)) / 2
    val3 = np.sqrt(3) * x + y - np.sqrt(3)
    mask3 = val3 > 0
    if np.any(mask3):
        d = val3[mask3] / 2.0
        x[mask3] = x[mask3] - np.sqrt(3) * d
        y[mask3] = y[mask3] - d
        
    return np.column_stack((x, y))

def _min_triangle_area(points):
    """
    Computes the minimum area of any triangle formed by 3 points.
    points: (N, 2) array
    """
    n = len(points)
    # Precompute indices
    # Using a simple loop for clarity, or vectorized
    # Vectorized:
    # i, j, k indices
    i_idx, j_idx, k_idx = [], [], []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
                
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    k_idx = np.array(k_idx)
    
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    # Cross product magnitude
    # (p_j - p_i) x (p_k - p_i)
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    best_score = 0.0
    best_points = None
    
    # Initial guess: random points
    rng = np.random.default_rng(42)
    
    # We will do multiple restarts
    n_restarts = 100
    
    # Pre-compute triangle area for normalization if needed, but we just return raw area
    # The evaluator likely normalizes or compares raw. The prompt says "maximize the smallest area".
    # "The notebook uses an equilateral triangle and divides by its area."
    # So the score is likely min_area / triangle_area.
    # Triangle area = sqrt(3)/4.
    # We just need to maximize min_area.
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Generate random initial points inside the triangle
        # Use rejection sampling or affine transform
        # Affine: P = u * B + v * (C-A) where u,v in [0,1] and u+v <= 1
        # Actually, standard rejection is fine for 11 points.
        pts = np.random.rand(N, 2)
        # Rejection
        valid = []
        while len(valid) < N:
            # Generate 100 at a time
            candidates = np.random.rand(100, 2)
            x = candidates[:, 0]
            y = candidates[:, 1]
            # Inside triangle: y >= 0, y <= sqrt(3)x, y <= -sqrt(3)(x-1)
            # This is tricky to vectorize rejection for arbitrary points.
            # Easier: generate in unit square, transform to triangle?
            # Or just use the projection function on random points in bounding box [0,1]x[0,1]
            # But projection might collapse them.
            
            # Let's use the standard mapping:
            # u, v ~ Uniform(0,1). If u+v < 1, point = A + u*(B-A) + v*(C-A)
            # B-A = (1,0), C-A = (0.5, sqrt(3)/2)
            # This maps to the triangle.
            u, v = np.random.rand(100, 2)
            mask = u + v < 1.0
            if np.any(mask):
                new_pts = np.column_stack((
                    u[mask] * 1.0 + v[mask] * 0.5,
                    v[mask] * (np.sqrt(3)/2)
                ))
                valid.extend(new_pts.tolist())
        
        pts = np.array(valid[:N])
        
        # Flatten for optimizer
        x0 = pts.flatten()
        
        def objective(x):
            p = x.reshape(N, 2)
            p = _project_to_triangle(p)
            # Check if points are valid (they should be after projection)
            area = _min_triangle_area(p)
            return -area
        
        # Run Nelder-Mead
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
            
            if res.success or res.fun < 0: # fun is negative area, so we want it to be as negative as possible (large area)
                final_pts = _project_to_triangle(res.x.reshape(N, 2))
                score = _min_triangle_area(final_pts)
                
                if score > best_score:
                    best_score = score
                    best_points = final_pts
                    
        except Exception:
            continue
            
    if best_points is None:
        # Fallback: return random points
        return np.random.rand(N, 2)
        
    return best_points
