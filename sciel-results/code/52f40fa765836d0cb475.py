import numpy as np
from scipy.optimize import minimize
import time

N = 11
# Reference triangle area
A_REF = np.sqrt(3) / 4.0

def compute_areas(params):
    """
    params: array of shape (N, 2), containing (u, v) for each point.
    Returns array of triangle areas for all combinations of 3 points.
    """
    u = params[:, 0]
    v = params[:, 1]
    w = 1.0 - u - v
    
    # Points in barycentric coords (u, v, w)
    # Area of triangle formed by points i, j, k is |det([u_i, v_i, w_i; ...])| * A_REF
    # We need to compute this for all i < j < k
    
    areas = []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                # Matrix rows are points i, j, k
                # Det of [[u_i, v_i, w_i], [u_j, v_j, w_j], [u_k, v_k, w_k]]
                # Since w = 1-u-v, we can simplify or just compute directly.
                # Direct computation is clear.
                u_i, v_i, w_i = u[i], v[i], w[i]
                u_j, v_j, w_j = u[j], v[j], w[j]
                u_k, v_k, w_k = u[k], v[k], w[k]
                
                det = (u_i * (v_j * w_k - w_j * v_k) -
                       v_i * (u_j * w_k - w_j * u_k) +
                       w_i * (u_j * v_k - v_j * u_k))
                area = abs(det) * A_REF
                areas.append(area)
                
    return np.array(areas)

def objective(params, beta, lambda_pen):
    """
    Minimize this function.
    """
    u = params[:, 0]
    v = params[:, 1]
    
    # Penalty for u+v > 1 (points outside triangle)
    # Use softplus for smoothness
    # x = u + v - 1
    x = u + v - 1.0
    # softplus(x) = log(1 + exp(x))
    # For large x, softplus(x) ~ x. For small x, ~ log(2).
    # We want to penalize x > 0.
    # A common smooth approximation to max(0, x) is log(1+exp(x)) - log(2) is not quite right for large x.
    # Let's use: p = (x + np.sqrt(x**2 + 1e-6)) / 2.0
    # This is smooth and approximates max(0, x).
    pen = np.sum((x + np.sqrt(x**2 + 1e-8)) / 2.0)
    
    areas = compute_areas(params)
    
    # Soft-min of areas: -1/beta * log(sum(exp(-beta * areas)))
    # To avoid overflow/underflow, subtract max
    max_area = np.max(areas)
    # If areas are very small, this might be unstable.
    # Let's add a small epsilon to areas to avoid log(0) or issues.
    areas_safe = areas + 1e-12
    
    log_sum_exp = np.max(-beta * areas_safe) + np.log(np.sum(np.exp(-beta * areas_safe - np.max(-beta * areas_safe))))
    soft_min = -1.0 / beta * log_sum_exp
    
    return soft_min + lambda_pen * pen

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin
    
    best_params = None
    best_score = -np.inf
    
    # Pre-compute combinations for speed? 
    # The loop in compute_areas is 165 iterations. 
    # N=11, so 11*10*9/6 = 165.
    # This is fast in Python if vectorized? 
    # Let's try to vectorize compute_areas.
    
    def compute_areas_vec(params):
        u = params[:, 0]
        v = params[:, 1]
        w = 1.0 - u - v
        
        # We want det for all i<j<k
        # This is hard to vectorize simply without itertools.
        # Given N=11, the loop is acceptable.
        # But we can optimize by pre-computing indices.
        indices = [(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)]
        
        areas = np.empty(len(indices))
        for idx, (i, j, k) in enumerate(indices):
            u_i, v_i, w_i = u[i], v[i], w[i]
            u_j, v_j, w_j = u[j], v[j], w[j]
            u_k, v_k, w_k = u[k], v[k], w[k]
            
            det = (u_i * (v_j * w_k - w_j * v_k) -
                   v_i * (u_j * w_k - w_j * u_k) +
                   w_i * (u_j * v_k - v_j * u_k))
            areas[idx] = abs(det) * A_REF
            
        return areas

    # Replace compute_areas with vectorized version in objective
    def objective_vec(params, beta, lambda_pen):
        u = params[:, 0]
        v = params[:, 1]
        
        x = u + v - 1.0
        pen = np.sum((x + np.sqrt(x**2 + 1e-8)) / 2.0)
        
        areas = compute_areas_vec(params)
        areas_safe = areas + 1e-12
        
        max_neg_beta_area = np.max(-beta * areas_safe)
        log_sum_exp = max_neg_beta_area + np.log(np.sum(np.exp(-beta * areas_safe - max_neg_beta_area)))
        soft_min = -1.0 / beta * log_sum_exp
        
        return soft_min + lambda_pen * pen

    # Initialization
    # Use a few random restarts
    for restart in range(20):
        if time.time() - start_time > timeout:
            break
            
        # Random init in barycentric coords
        # u, v >= 0, u+v <= 1
        # Sample u, v uniformly? 
        # Simple rejection:
        u = np.random.rand(N)
        v = np.random.rand(N)
        mask = u + v > 1
        while np.any(mask):
            u[mask] = np.random.rand(np.sum(mask))
            v[mask] = np.random.rand(np.sum(mask))
            mask = u + v > 1
            
        params = np.column_stack((u, v))
        
        # Optimization
        # Start with moderate beta and lambda
        beta = 10.0
        lambda_pen = 100.0
        
        # Run L-BFGS-B
        # Bounds for u, v are [0, 1]
        bounds = [(0.0, 1.0) for _ in range(2 * N)]
        
        try:
            res = minimize(
                objective_vec, 
                params, 
                method='L-BFGS-B', 
                bounds=bounds, 
                args=(beta, lambda_pen),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.success:
                p_opt = res.x
                # Evaluate actual min area
                areas = compute_areas_vec(p_opt)
                min_area = np.min(areas)
                
                if min_area > best_score:
                    best_score = min_area
                    best_params = p_opt.copy()
                    
        except Exception:
            pass

    if best_params is None:
        # Fallback to random
        u = np.random.rand(N)
        v = np.random.rand(N)
        mask = u + v > 1
        while np.any(mask):
            u[mask] = np.random.rand(np.sum(mask))
            v[mask] = np.random.rand(np.sum(mask))
            mask = u + v > 1
        best_params = np.column_stack((u, v))

    # Convert barycentric (u, v) to Cartesian (x, y)
    # x = u * 0 + v * 1 + w * 0.5 = v + 0.5 * (1 - u - v) = 0.5 - 0.5u + 0.5v
    # y = u * 0 + v * 0 + w * sqrt(3)/2 = 0.5 * sqrt(3) * (1 - u - v)
    
    u = best_params[:, 0]
    v = best_params[:, 1]
    w = 1.0 - u - v
    
    x = 0.5 - 0.5 * u + 0.5 * v
    y = np.sqrt(3) / 2.0 * w
    
    return np.column_stack((x, y))
