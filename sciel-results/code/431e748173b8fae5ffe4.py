import numpy as np
from itertools import combinations
import time
from scipy.optimize import minimize

N = 11
TIME_LIMIT = 140.0  # Safety margin for 150s limit

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

def bary_to_cart(bary):
    """Convert barycentric coords (u, v) to Cartesian (x, y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def calc_areas_bary(bary):
    """Calculate areas of all triangles formed by N points in barycentric coords."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    ui = u[tri_idx[:, 0]]
    vi = v[tri_idx[:, 0]]
    wi = w[tri_idx[:, 0]]
    
    uj = u[tri_idx[:, 1]]
    vj = v[tri_idx[:, 1]]
    wj = w[tri_idx[:, 1]]
    
    uk = u[tri_idx[:, 2]]
    vk = v[tri_idx[:, 2]]
    wk = w[tri_idx[:, 2]]
    
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    return np.abs(det)

def clamp_bary(bary, eps=1e-9):
    """Clamp barycentric coordinates to the strictly valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    return np.column_stack([u, v])

def get_min_area(bary):
    """Get the minimum area for a given barycentric configuration."""
    areas = calc_areas_bary(bary)
    return np.min(areas)

def soft_min_obj(bary, beta=20.0):
    """
    Objective function for minimization: -soft_min(areas).
    soft_min(x) = -1/beta * log(sum(exp(-beta * x)))
    Minimizing -soft_min is equivalent to maximizing soft_min.
    As beta -> infinity, soft_min -> min.
    """
    areas = calc_areas_bary(bary)
    # Numerical stability: shift areas by max area
    max_a = np.max(areas)
    exp_vals = np.exp(-beta * (areas - max_a))
    log_sum = np.log(np.sum(exp_vals)) + (-beta * max_a)
    soft_min_val = -1.0 / beta * log_sum
    return -soft_min_val, soft_min_val

def generate_hexagonal_init():
    """Generate a hexagonal lattice-like initialization."""
    pts = []
    scale = 0.3
    max_k = int(1.0 / scale) + 1
    for k in range(max_k + 1):
        w_val = k * scale
        if w_val > 1.0:
            break
        num_pts_in_row = int((1 - w_val) / scale) + 1
        for j in range(num_pts_in_row):
            u_val = j * scale
            v_val = (1 - w_val) - u_val
            if u_val >= -1e-9 and v_val >= -1e-9 and u_val + v_val <= 1 + 1e-9:
                pts.append([u_val, v_val])
    if len(pts) >= N:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pts), N, replace=False)
        return np.array(pts)[idx]
    return None

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_val
        if bary is None: return
        bary = clamp_bary(bary)
        val = get_min_area(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Initialization
    start_points = []
    
    # Hexagonal
    hex_init = generate_hexagonal_init()
    if hex_init is not None:
        start_points.append(hex_init)
        
    # Random starts
    rng_master = np.random.default_rng(123)
    for _ in range(10):
        u = rng_master.random(N)
        v = rng_master.random(N) * (1 - u)
        start_points.append(np.column_stack([u, v]))
        
    # Symmetric/Structured starts (approximate Fermat points or grid)
    # A simple grid in barycentric coords
    grid_u = np.linspace(0.1, 0.9, 4)
    grid_v = np.linspace(0.1, 0.9, 4)
    grid_pts = []
    for u in grid_u:
        for v in grid_v:
            if u + v < 0.9:
                grid_pts.append([u, v])
    if len(grid_pts) >= N:
        rng_grid = np.random.default_rng(99)
        idx = rng_grid.choice(len(grid_pts), N, replace=False)
        start_points.append(np.array(grid_pts)[idx])

    # 2. Optimization Loop
    # We will run scipy.optimize.minimize on each start point.
    # We use L-BFGS-B with bounds.
    # Bounds: u in [0, 1], v in [0, 1], u+v <= 1.
    # L-BFGS-B handles box constraints well. The u+v<=1 constraint is handled by clamping 
    # inside the objective or by careful bounds. 
    # Since barycentric coords u,v >=0 and u+v<=1, we can define bounds:
    # u_i in [0, 1], v_i in [0, 1].
    # The constraint u+v<=1 is non-box. 
    # To strictly satisfy u+v<=1, we can parameterize or just clamp in the objective.
    # Clamping in objective makes it non-smooth at boundaries. 
    # Alternative: Use variables in a simpler space? 
    # Let's stick to barycentric and clamp. The gradient might be slightly off at boundaries 
    # but L-BFGS-B is robust.
    
    bounds = [(0.0, 1.0) for _ in range(N * 2)]
    
    # We will iterate through start points and run optimization.
    # To ensure we finish in time, we limit the number of function evaluations per run.
    
    num_starts = len(start_points)
    max_fevals = 500  # L-BFGS-B is fast, 500 evals is usually enough for local convergence
    
    for i, sp in enumerate(start_points):
        if time.time() > deadline - 10:
            break
            
        # Flatten barycentric coords to 1D array for scipy
        x0 = sp.flatten()
        
        # Define the objective function for scipy
        # It needs to return scalar. We use soft_min with a moderate beta.
        # High beta makes it closer to true min but harder to optimize.
        # We can anneal beta, but let's start with a fixed reasonable beta.
        beta = 10.0
        
        def objective(x):
            bary_2d = x.reshape(N, 2)
            # Clamp to valid region to handle boundary issues during optimization
            bary_clamped = clamp_bary(bary_2d)
            val, _ = soft_min_obj(bary_clamped, beta=beta)
            return val
            
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 100, 'maxfev': max_fevals})
            
            if res.success or res.fun < 1e10: # Check if it didn't crash
                best_bary_res = res.x.reshape(N, 2)
                update_best(best_bary_res)
                
        except Exception:
            pass

    # 3. Refinement Phase
    # If we have a good candidate, try to refine it with higher beta (sharper min)
    if best_bary is not None and time.time() < deadline - 5:
        # Run a few more optimizations on the best found with higher beta
        for beta_refine in [20.0, 50.0, 100.0]:
            if time.time() > deadline - 2:
                break
            
            x0 = best_bary.flatten()
            
            def objective_refine(x):
                bary_2d = x.reshape(N, 2)
                bary_clamped = clamp_bary(bary_2d)
                val, _ = soft_min_obj(bary_clamped, beta=beta_refine)
                return val
                
            try:
                res = minimize(objective_refine, x0, method='L-BFGS-B', bounds=bounds, 
                               options={'maxiter': 50, 'maxfev': 200})
                if res.success or res.fun < 1e10:
                    best_bary_res = res.x.reshape(N, 2)
                    update_best(best_bary_res)
            except Exception:
                pass

    # 4. Final Fallback
    if best_bary is None:
        rng = np.random.default_rng(0)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
        
    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
