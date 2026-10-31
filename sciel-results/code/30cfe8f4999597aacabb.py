import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import combinations
import time

N = 11
TIME_LIMIT = 115.0

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

def clamp_bary(bary, eps=1e-10):
    """Clamp barycentric coordinates to the valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    # Ensure u >= 0, v >= 0, u + v <= 1
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.maximum(sums, eps)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    return np.column_stack([u, v])

def hard_min_value(bary):
    """Compute the hard minimum area value."""
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return np.min(areas)

def active_set_refine(bary_init, time_limit=10.0):
    """
    Active set method: identify binding constraints and optimize.
    """
    start_time = time.time()
    bary = bary_init.copy()
    bounds = [(0, 1)] * (N * 2)
    
    for iteration in range(50):
        if time.time() - start_time > time_limit:
            break
        
        areas = calc_areas_bary(bary)
        min_area = np.min(areas)
        
        # Find all triangles at (or near) the minimum
        tol = 1e-8
        active_mask = np.abs(areas - min_area) < tol
        active_indices = np.where(active_mask)[0]
        
        if len(active_indices) == 0:
            active_indices = np.array([np.argmin(areas)])
        
        def obj_and_grad(x):
            b = clamp_bary(x.reshape(N, 2))
            a = calc_areas_bary(b)
            a_active = a[active_indices]
            min_idx = np.argmin(a_active)
            min_val = a_active[min_idx]
            return -min_val
        
        try:
            result = minimize(
                obj_and_grad,
                bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14}
            )
            if -result.fun > hard_min_value(bary):
                bary = clamp_bary(result.x.reshape(N, 2))
        except Exception:
            pass
    
    return bary

def differential_evolution_search(time_limit=60.0):
    """
    Global optimization using differential evolution.
    """
    start_time = time.time()
    
    bounds = [(0, 1)] * (N * 2)
    
    def de_obj(x):
        b = x.reshape(N, 2)
        b = clamp_bary(b)
        return -hard_min_value(b)
    
    try:
        result = differential_evolution(
            de_obj,
            bounds,
            maxiter=500,
            popsize=20,
            tol=1e-12,
            mutation=(0.5, 1.0),
            recombination=0.8,
            seed=42,
            polish=True,
            init='latinhypercube'
        )
        best_bary = clamp_bary(result.x.reshape(N, 2))
    except Exception:
        best_bary = None
    
    return best_bary

def generate_good_init():
    """
    Generate a good initial configuration based on known optimal patterns.
    """
    best_bary = None
    best_val = -np.inf
    
    for scale in [0.2, 0.25, 0.3, 0.35]:
        pts = []
        max_row = int(1.0 / scale) + 1
        for row in range(max_row):
            w_val = row * scale
            if w_val > 1.0:
                break
            num_in_row = int((1 - w_val) / scale) + 1
            for col in range(num_in_row):
                u_val = col * scale
                v_val = (1 - w_val) - u_val
                if u_val >= -1e-9 and v_val >= -1e-9 and u_val + v_val <= 1 + 1e-9:
                    pts.append([u_val, v_val])
        
        if len(pts) >= N:
            rng = np.random.default_rng(42)
            for trial in range(20):
                if len(pts) == N:
                    bary = np.array(pts)
                else:
                    idx = rng.choice(len(pts), N, replace=False)
                    bary = np.array(pts)[idx]
                val = hard_min_value(bary)
                if val > best_val:
                    best_val = val
                    best_bary = bary.copy()
    
    if best_bary is None:
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
    
    return best_bary

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_val
        if bary is None:
            return
        bary = clamp_bary(bary)
        val = hard_min_value(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # Step 1: Generate good initial configuration
    init_bary = generate_good_init()
    update_best(init_bary)
    
    # Step 2: Differential evolution for global search
    if time.time() < deadline - 30:
        de_time = min(45.0, deadline - time.time() - 30)
        de_result = differential_evolution_search(time_limit=de_time)
        if de_result is not None:
            update_best(de_result)
    
    # Step 3: Active set refinement on best solution
    if best_bary is not None and time.time() < deadline - 10:
        active_time = min(15.0, deadline - time.time() - 10)
        refined = active_set_refine(best_bary, time_limit=active_time)
        update_best(refined)
    
    # Step 4: Additional local refinement with Nelder-Mead
    if best_bary is not None and time.time() < deadline - 5:
        try:
            def hard_obj(x):
                return -hard_min_value(x.reshape(N, 2))
            
            result_nm = minimize(
                hard_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
            )
            update_best(clamp_bary(result_nm.x.reshape(N, 2)))
        except Exception:
            pass
    
    # Step 5: Final SLSQP polish
    if best_bary is not None and time.time() < deadline - 2:
        try:
            bounds = [(0, 1)] * (N * 2)
            def hard_obj(x):
                return -hard_min_value(x.reshape(N, 2))
            
            result_slsqp = minimize(
                hard_obj,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15}
            )
            update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
        except Exception:
            pass
    
    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
