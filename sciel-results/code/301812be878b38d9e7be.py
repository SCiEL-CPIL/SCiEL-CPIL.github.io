import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 11
H = np.sqrt(3) / 2

# Vertices of the equilateral triangle
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, H])

# Precompute indices for vectorized area calculation
idx_i, idx_j, idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            idx_i.append(i)
            idx_j.append(j)
            idx_k.append(k)

idx_i = np.array(idx_i)
idx_j = np.array(idx_j)
idx_k = np.array(idx_k)

def barycentric_to_cartesian(params):
    """
    Convert barycentric-like coordinates (u, v) to Cartesian.
    params: (N, 2) array of (u, v).
    Point = u*A + v*B + (1-u-v)*C
    We enforce u>=0, v>=0, u+v<=1 by clipping.
    """
    u = params[:, 0]
    v = params[:, 1]
    
    # Clip to ensure valid barycentric coordinates
    # If u+v > 1, scale down
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
        
    # Ensure non-negative
    u = np.maximum(u, 0.0)
    v = np.maximum(v, 0.0)
    
    w = 1.0 - u - v
    
    # Cartesian coordinates
    x = u * B[0] + v * C[0] + w * A[0]
    y = u * B[1] + v * C[1] + w * A[1]
    
    return np.column_stack((x, y))

def min_triangle_area_cartesian(points):
    """
    points: (N, 2) array in Cartesian coordinates.
    Returns the minimum area of any triangle formed by 3 points.
    """
    p1 = points[idx_i]
    p2 = points[idx_j]
    p3 = points[idx_k]
    
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def objective(params):
    """
    Objective function for minimization.
    params: flattened array of (u, v) for N points.
    Returns -min_area.
    """
    bary = params.reshape(N, 2)
    points = barycentric_to_cartesian(bary)
    area = min_triangle_area_cartesian(points)
    return -area

def refine_local(params, deadline):
    """
    Perform local refinement using Powell's method.
    params: flattened (N, 2) array.
    """
    best_params = params.copy()
    best_score = -objective(best_params)
    
    rng = np.random.default_rng(42)
    
    # Try multiple restarts with Powell
    for attempt in range(20):
        if time.time() > deadline:
            break
            
        if attempt == 0:
            x0 = best_params.copy()
        else:
            # Perturb in barycentric space
            # Small noise
            scale = 0.01 * (attempt + 1)
            noise = rng.normal(0, scale, size=(N, 2))
            x0 = (best_params.reshape(N, 2) + noise).flatten()
            
        try:
            res = minimize(
                objective,
                x0,
                method='Powell',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            new_params = res.x
            new_score = -objective(new_params)
            
            if new_score > best_score:
                best_score = new_score
                best_params = new_params.copy()
                
        except Exception:
            continue
            
    return best_params

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Bounds for barycentric coordinates u, v in [0, 1]
    # The clipping in barycentric_to_cartesian handles the u+v<=1 constraint
    bounds = [(0.0, 1.0), (0.0, 1.0)] * N
    
    best_params = None
    best_score = -float('inf')
    
    # 1. Global Search with Differential Evolution
    try:
        result = differential_evolution(
            objective,
            bounds,
            maxiter=200, 
            popsize=15,
            tol=0.01,
            seed=42,
            polish=False,
            init='sobol' 
        )
        
        candidate_params = result.x
        candidate_score = -objective(candidate_params)
        
        if candidate_score > best_score:
            best_score = candidate_score
            best_params = candidate_params.copy()
            
    except Exception:
        pass
        
    # Fallback if DE fails
    if best_params is None:
        rng = np.random.default_rng(42)
        best_params = rng.random((N, 2)).flatten()
        best_score = -objective(best_params)
    
    # 2. Local Refinement
    if time.time() < deadline - 5:
        best_params = refine_local(best_params, deadline)
        
    # 3. Final check and return
    bary = best_params.reshape(N, 2)
    points = barycentric_to_cartesian(bary)
    
    # Ensure points are strictly inside or on boundary
    # The clipping handles this, but let's double check for numerical errors
    return points

