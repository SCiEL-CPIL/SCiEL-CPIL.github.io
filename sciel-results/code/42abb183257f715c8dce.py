"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Vertices of the unit equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = 0.5 * np.sqrt(3) / 2  # Area of the container triangle

def _bary_to_cart(bary_coords):
    """
    Convert barycentric coordinates (u, v) to Cartesian (x, y).
    w = 1 - u - v
    P = u*V0 + v*V1 + w*V2
    """
    u = bary_coords[:, 0]
    v = bary_coords[:, 1]
    w = 1.0 - u - v
    x = 0.5 + 0.5 * v - 0.5 * u
    y = (np.sqrt(3) / 2) * (1.0 - u - v)
    return np.column_stack((x, y))

def _min_triangle_area_vectorized(bary_points):
    """
    Calculate the minimum area of all triangles formed by N points.
    bary_points: (N, 2) array of (u, v) coordinates.
    """
    cart = _bary_to_cart(bary_points)
    N_pts = len(cart)
    
    i_idx, j_idx, k_idx = np.indices((N_pts, N_pts, N_pts))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    p_i = cart[i_flat]
    p_j = cart[j_flat]
    p_k = cart[k_flat]
    
    dx1 = p_j[:, 0] - p_i[:, 0]
    dy1 = p_j[:, 1] - p_i[:, 1]
    dx2 = p_k[:, 0] - p_i[:, 0]
    dy2 = p_k[:, 1] - p_i[:, 1]
    
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _objective_bary(x, beta=10.0):
    """
    Objective function to minimize: Negative of the smooth minimum area.
    """
    bary_points = x.reshape(N, 2)
    u = bary_points[:, 0]
    v = bary_points[:, 1]
    
    # Calculate all triangle areas
    cart = _bary_to_cart(bary_points)
    N_pts = len(cart)
    
    i_idx, j_idx, k_idx = np.indices((N_pts, N_pts, N_pts))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    p_i = cart[i_flat]
    p_j = cart[j_flat]
    p_k = cart[k_flat]
    
    dx1 = p_j[:, 0] - p_i[:, 0]
    dy1 = p_j[:, 1] - p_i[:, 1]
    dx2 = p_k[:, 0] - p_i[:, 0]
    dy2 = p_k[:, 1] - p_i[:, 1]
    
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    
    # Smooth minimum using log-sum-exp
    max_area = np.max(areas)
    log_sum_exp = np.log(np.sum(np.exp(-beta * (areas - max_area)))) - beta * max_area
    smooth_min = -log_sum_exp / beta
    
    return -smooth_min

def _make_structured_starts():
    """Generate diverse starting configurations in barycentric coordinates."""
    starts = []
    
    # 1. Grid based
    grid_u = np.linspace(0.1, 0.8, 10)
    grid_v = np.linspace(0.1, 0.8, 10)
    valid_points = []
    for u in grid_u:
        for v in grid_v:
            if u + v <= 0.9:
                valid_points.append([u, v])
                
    rng = np.random.RandomState(42)
    if len(valid_points) >= N:
        indices = rng.choice(len(valid_points), N, replace=False)
        starts.append(np.array(valid_points)[indices])
        
    # 2. Random starts
    for seed in [1, 2, 3, 4, 5]:
        rng = np.random.RandomState(seed)
        pts = []
        while len(pts) < N:
            u = rng.random()
            v = rng.random()
            if u + v < 1.0:
                pts.append([u, v])
        starts.append(np.array(pts))
        
    return starts

def _constraints_for_slsqp():
    """Define constraints for SLSQP."""
    cons = []
    for i in range(N):
        # u_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i]})
        # v_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1]})
        # u_i + v_i <= 1  =>  1 - u_i - v_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*i+1]})
    return cons

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_x = None
    best_score = -float('inf')
    
    starts = _make_structured_starts()
    constraints = _constraints_for_slsqp()
    
    # Annealing beta
    betas = [10.0, 50.0, 100.0, 200.0, 500.0]
    
    current_x = None
    
    for beta in betas:
        if time.time() > deadline:
            break
            
        if current_x is None:
            # Pick best start
            best_start = None
            best_start_score = -float('inf')
            for s in starts:
                s_flat = s.flatten()
                score = _min_triangle_area_vectorized(s)
                if score > best_start_score:
                    best_start_score = score
                    best_start = s_flat
            current_x = best_start if best_start is not None else starts[0].flatten()
            
        try:
            res = minimize(
                _objective_bary, 
                current_x, 
                method='SLSQP', 
                constraints=constraints, 
                options={'maxiter': 100, 'ftol': 1e-8, 'eps': 1e-5}
            )
            current_x = res.x
        except Exception:
            # If SLSQP fails, try L-BFGS-B with penalty
            pass

        x_opt_bary = current_x.reshape(N, 2)
        # Ensure validity
        u = x_opt_bary[:, 0]
        v = x_opt_bary[:, 1]
        if np.any(u < -1e-6) or np.any(v < -1e-6) or np.any(u + v > 1.0 + 1e-6):
            # Project back
            # Simple clamp
            u = np.clip(u, 0, 1)
            v = np.clip(v, 0, 1)
            # If u+v > 1, scale down
            sum_uv = u + v
            mask_out = sum_uv > 1.0
            if np.any(mask_out):
                scale = 1.0 / sum_uv[mask_out]
                u[mask_out] *= scale
                v[mask_out] *= scale
            x_opt_bary = np.column_stack((u, v))
            
        score = _min_triangle_area_vectorized(x_opt_bary)
        
        if score > best_score:
            best_score = score
            best_x = x_opt_bary.copy()

    if best_x is None:
        best_x = starts[0]
        
    return _bary_to_cart(best_x)

# EVOLVE-BLOCK-END
