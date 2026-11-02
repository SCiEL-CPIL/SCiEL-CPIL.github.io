import numpy as np
import time
from scipy.optimize import minimize

N = 11
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3)/2])

def project_points_to_triangle(points):
    pts = points.copy()
    x = pts[:, 0]
    y = pts[:, 1]
    
    # 1. y < 0
    mask_ab = y < 0
    if np.any(mask_ab):
        pts[mask_ab, 1] = 0
        pts[mask_ab, 0] = np.clip(pts[mask_ab, 0], 0, 1)
        x = pts[:, 0]
        y = pts[:, 1]

    # 2. y > sqrt(3) * x
    mask_ac = y > np.sqrt(3) * x
    if np.any(mask_ac):
        x_ac = x[mask_ac] / 4 + np.sqrt(3) * y[mask_ac] / 4
        y_ac = np.sqrt(3) * x[mask_ac] / 4 + 3 * y[mask_ac] / 4
        t = 2 * x_ac
        valid = (t >= 0) & (t <= 1)
        
        res_x = np.zeros_like(x_ac)
        res_y = np.zeros_like(y_ac)
        res_x[valid] = x_ac[valid]
        res_y[valid] = y_ac[valid]
        
        mask_a = ~valid & (t < 0)
        mask_c = ~valid & (t > 1)
        res_x[mask_a] = 0
        res_y[mask_a] = 0
        res_x[mask_c] = 0.5
        res_y[mask_c] = np.sqrt(3)/2
        
        pts[mask_ac, 0] = res_x
        pts[mask_ac, 1] = res_y
        x = pts[:, 0]
        y = pts[:, 1]

    # 3. y > sqrt(3) * (1 - x)
    mask_bc = y > np.sqrt(3) * (1 - x)
    if np.any(mask_bc):
        x_bc = x[mask_bc] / 4 - np.sqrt(3) * y[mask_bc] / 4 + 3 / 4
        y_bc = 3 * y[mask_bc] / 4 - np.sqrt(3) * x[mask_bc] / 4 + np.sqrt(3) / 4
        t = 2 * (1 - x_bc)
        valid = (t >= 0) & (t <= 1)
        
        res_x = np.zeros_like(x_bc)
        res_y = np.zeros_like(y_bc)
        res_x[valid] = x_bc[valid]
        res_y[valid] = y_bc[valid]
        
        mask_b = ~valid & (t < 0)
        mask_c = ~valid & (t > 1)
        res_x[mask_b] = 1
        res_y[mask_b] = 0
        res_x[mask_c] = 0.5
        res_y[mask_c] = np.sqrt(3)/2
        
        pts[mask_bc, 0] = res_x
        pts[mask_bc, 1] = res_y

    return pts

def compute_min_area_vectorized(points):
    n = len(points)
    i, j, k = np.indices((n, n, n))
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    p_i = points[i]
    p_j = points[j]
    p_k = points[k]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def objective(flat_points):
    points = flat_points.reshape(N, 2)
    points = project_points_to_triangle(points)
    val = compute_min_area_vectorized(points)
    return -val

def construct():
    start_time = time.time()
    time_limit = 120
    deadline = start_time + time_limit - 5
    
    best_score = -1
    best_points = None
    
    def random_init():
        pts = np.random.rand(N, 2)
        pts[:, 1] *= np.sqrt(3)/2
        return project_points_to_triangle(pts)

    # Initializations
    inits = [random_init() for _ in range(10)]
    
    # Add a grid init
    # 3x4 grid scaled?
    # Let's just use randoms.
    
    num_restarts = 0
    max_restarts = 100 # We can afford more now
    
    while time.time() < deadline:
        if num_restarts < len(inits):
            x0 = inits[num_restarts]
        else:
            x0 = random_init()
            
        x0_flat = x0.flatten()
        
        try:
            res = minimize(objective, x0_flat, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
            
            pts = res.x.reshape(N, 2)
            pts = project_points_to_triangle(pts)
            score = compute_min_area_vectorized(pts)
            
            if score > best_score:
                best_score = score
                best_points = pts
                
        except Exception:
            pass
            
        num_restarts += 1
        
        if time.time() > deadline - 10:
            break

    if best_points is None:
        best_points = random_init()
        
    return best_points
