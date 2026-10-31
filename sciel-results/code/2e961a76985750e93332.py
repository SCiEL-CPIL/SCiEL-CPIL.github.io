import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 145.0  # Safety margin for 150s limit

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

def hard_min_obj(bary_flat):
    """Objective using hard min."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def soft_min_obj(bary_flat, beta=50.0):
    """
    Smooth surrogate for the hard min via log-sum-exp.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    if np.any(areas < 1e-15):
        return 1e10
    
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    
    if max_scaled > 700:
        return -np.min(areas)
        
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    return -soft_min

def sym_params_to_bary(params):
    """
    Convert symmetry parameters to barycentric coordinates.
    
    Parameters (10 values):
    - params[0]: center point u
    - params[1]: center point v
    - params[2]: radius for inner ring (3 points at 120 deg)
    - params[3]: phase for inner ring
    - params[4]: radius for outer ring (3 points at 120 deg)
    - params[5]: phase for outer ring
    - params[6]: vertex offset (how far from vertices, 0=at vertex, 1=toward center)
    - params[7]: additional radial scale for vertex points
    - params[8]: shear x component
    - params[9]: shear y component
    
    The 3-fold symmetry means:
    - 1 center point
    - 3 points on inner ring (120 deg apart)
    - 3 points on outer ring (120 deg apart)
    - 3 points near vertices (with offset)
    Total: 10 points... need 1 more.
    
    Let me reconsider: 1 center + 3 inner + 3 outer + 3 vertex + 1 extra = 11
    """
    bary = np.zeros((N, 2))
    
    # Center point (index 0)
    bary[0] = [params[0], params[1]]
    
    # Inner ring: 3 points at 120 deg intervals (indices 1-3)
    r_inner = params[2]
    phase_inner = params[3]
    for i in range(3):
        angle = phase_inner + 2 * np.pi * i / 3
        du = r_inner * np.cos(angle)
        dv = r_inner * np.sin(angle)
        bary[1 + i] = [1/3 + du, 1/3 + dv]
    
    # Outer ring: 3 points at 120 deg intervals (indices 4-6)
    r_outer = params[4]
    phase_outer = params[5]
    for i in range(3):
        angle = phase_outer + 2 * np.pi * i / 3
        du = r_outer * np.cos(angle)
        dv = r_outer * np.sin(angle)
        bary[4 + i] = [1/3 + du, 1/3 + dv]
    
    # Vertex points: 3 points near vertices (indices 7-9)
    # Vertex 0 is at barycentric (1,0), Vertex 1 at (0,1), Vertex 2 at (0,0)
    # Offset toward center by factor (1 - vertex_offset)
    vo = params[6]  # 0 = at vertex, 1 = at center
    vs = params[7]  # scale factor
    
    # Vertex 0: (1,0) -> offset toward center (1/3, 1/3)
    bary[7] = [1 - vo*(1 - 1/3)*vs, 0 + vo*(1/3)*vs]
    # Vertex 1: (0,1) -> offset toward center (1/3, 1/3)
    bary[8] = [0 + vo*(1/3)*vs, 1 - vo*(1 - 1/3)*vs]
    # Vertex 2: (0,0) -> offset toward center (1/3, 1/3)
    bary[9] = [0 + vo*(1/3)*vs, 0 + vo*(1/3)*vs]
    
    # Extra point (index 10): use shear to place it
    # Base position at (0.5, 0.25) with shear
    bary[10] = [0.5 + params[8], 0.25 + params[9]]
    
    return clamp_bary(bary)

def sym_obj(params):
    """Objective for symmetry parameterization."""
    bary = sym_params_to_bary(params)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def sym_soft_obj(params, beta=50.0):
    """Soft min objective for symmetry parameterization."""
    bary = sym_params_to_bary(params)
    areas = calc_areas_bary(bary)
    
    if np.any(areas < 1e-15):
        return 1e10
    
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    
    if max_scaled > 700:
        return -np.min(areas)
        
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    return -soft_min

def generate_symmetric_init():
    """
    Generate a symmetric configuration exploiting 3-fold symmetry.
    3 points near vertices, 6 near edge midpoints, 1 center, 1 additional.
    """
    bary = np.zeros((N, 2))
    
    # Near vertex 0: barycentric (1,0,0) -> (u=1,v=0)
    bary[0] = [0.95, 0.02]
    # Near vertex 1: barycentric (0,1,0) -> (u=0,v=1)
    bary[1] = [0.02, 0.95]
    # Near vertex 2: barycentric (0,0,1) -> (u=0,v=0)
    bary[2] = [0.02, 0.02]
    
    # Near edge midpoints (offset inward)
    bary[3] = [0.48, 0.03]  # edge 0-1
    bary[4] = [0.03, 0.48]  # edge 1-2
    bary[5] = [0.48, 0.48]  # edge 2-0
    
    # Center
    bary[6] = [1/3, 1/3]
    
    # Additional points: form a small hexagon around center
    r = 0.15
    for i in range(3):
        angle = 2 * np.pi * i / 3 + np.pi / 6
        du = r * np.cos(angle)
        dv = r * np.sin(angle)
        bary[7 + i] = [1/3 + du, 1/3 + dv]
    
    # One more point
    bary[10] = [0.5, 0.25]
    
    return clamp_bary(bary)

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration.
    """
    best_bary = None
    best_val = -np.inf
    
    # Try a range of scales to find a good starting point
    for scale in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
        pts = []
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
            # Try different subsets
            rng = np.random.default_rng(42)
            for _ in range(5):
                if len(pts) == N:
                    bary = np.array(pts)
                else:
                    idx = rng.choice(len(pts), N, replace=False)
                    bary = np.array(pts)[idx]
                val = -hard_min_obj(bary.flatten())
                if val > best_val:
                    best_val = val
                    best_bary = bary.copy()
                
    if best_bary is None:
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
        
    return best_bary

def generate_random_init(seed=None):
    """Generate random valid barycentric coordinates."""
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def generate_perturbed_init(base_bary, sigma=0.05, seed=None):
    """Generate a perturbed version of a base configuration."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, base_bary.shape)
    return clamp_bary(base_bary + noise)

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Graduated soft-min minimization (beta 20 -> 50 -> 200)
    for beta in [20.0, 50.0, 200.0]:
        if time.time() - start_time > time_limit * 0.35:
            break
        try:
            result1 = minimize(
                soft_min_obj,
                bary_current.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result1.success or result1.fun < -hard_min_obj(bary_current.flatten()):
                bary_current = clamp_bary(result1.x.reshape(N, 2))
        except Exception:
            pass
    
    if time.time() - start_time > time_limit * 0.4:
        return bary_current
        
    # Stage 2: Hard min refinement with Nelder-Mead
    try:
        result2 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1500, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
        )
        bary_current = clamp_bary(result2.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.8:
        return bary_current
    
    # Stage 3: SLSQP polish
    try:
        result3 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14}
        )
        bary_current = clamp_bary(result3.x.reshape(N, 2))
    except Exception:
        pass
        
    return bary_current

def refine_sym(params_init, time_limit=5.0):
    """
    Refine symmetry parameters using multi-stage optimization.
    """
    start_time = time.time()
    
    # Bounds for symmetry parameters
    sym_bounds = [
        (0.2, 0.8),   # center u
        (0.2, 0.8),   # center v
        (0.01, 0.3),  # inner radius
        (0, 2*np.pi), # inner phase
        (0.1, 0.4),   # outer radius
        (0, 2*np.pi), # outer phase
        (0.1, 0.8),   # vertex offset
        (0.5, 1.5),   # vertex scale
        (-0.1, 0.1),  # shear x
        (-0.1, 0.1),  # shear y
    ]
    
    params_current = params_init.copy()
    
    # Stage 1: Soft-min with increasing beta
    for beta in [30.0, 100.0, 300.0]:
        if time.time() - start_time > time_limit * 0.4:
            break
        try:
            result = minimize(
                sym_soft_obj,
                params_current,
                args=(beta,),
                method='L-BFGS-B',
                bounds=sym_bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result.success or result.fun < sym_obj(params_current):
                params_current = result.x
        except Exception:
            pass
    
    if time.time() - start_time > time_limit * 0.6:
        return params_current
    
    # Stage 2: Nelder-Mead on hard min
    try:
        result2 = minimize(
            sym_obj,
            params_current,
            method='Nelder-Mead',
            bounds=sym_bounds,
            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
        )
        params_current = result2.x
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.85:
        return params_current
    
    # Stage 3: SLSQP polish
    try:
        result3 = minimize(
            sym_obj,
            params_current,
            method='SLSQP',
            bounds=sym_bounds,
            options={'maxiter': 500, 'ftol': 1e-14}
        )
        params_current = result3.x
    except Exception:
        pass
    
    return params_current

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = -hard_min_obj(bary.flatten())
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Try Symmetry-parameterized search
    if time.time() < deadline - 60:
        # Initial symmetric parameters
        sym_init = np.array([1/3, 1/3, 0.15, np.pi/6, 0.25, np.pi/3, 0.3, 1.0, 0.0, 0.0])
        refined_sym_params = refine_sym(sym_init, time_limit=10.0)
        sym_bary = sym_params_to_bary(refined_sym_params)
        update_best(sym_bary)
        
        # Try multiple initial conditions for symmetry params
        rng = np.random.default_rng(42)
        for _ in range(5):
            if time.time() > deadline - 40:
                break
            sym_try = np.array([
                1/3 + rng.normal(0, 0.05),
                1/3 + rng.normal(0, 0.05),
                0.1 + rng.uniform(0, 0.2),
                rng.uniform(0, 2*np.pi),
                0.15 + rng.uniform(0, 0.2),
                rng.uniform(0, 2*np.pi),
                0.2 + rng.uniform(0, 0.4),
                0.8 + rng.uniform(0, 0.4),
                rng.normal(0, 0.03),
                rng.normal(0, 0.03)
            ])
            refined_try = refine_sym(sym_try, time_limit=4.0)
            try_bary = sym_params_to_bary(refined_try)
            update_best(try_bary)
    
    # 2. Try Hexagonal Init
    hex_init = generate_hexagonal_init()
    refined_hex = refine_local(hex_init, time_limit=8.0)
    update_best(refined_hex)
    
    # 2b. Try Symmetric Init (original)
    sym_init_orig = generate_symmetric_init()
    refined_sym_orig = refine_local(sym_init_orig, time_limit=8.0)
    update_best(refined_sym_orig)
    
    # 3. Basin Hopping with adaptive perturbation
    if time.time() < deadline - 20:
        rng = np.random.default_rng(123)
        bh_sigma = 0.08
        no_improve_count = 0
        max_no_improve = 10
        
        while time.time() < deadline - 15:
            if best_bary is None:
                init = generate_random_init(seed=5000)
            elif no_improve_count < max_no_improve:
                init = generate_perturbed_init(best_bary, sigma=bh_sigma, seed=6000 + int(time.time()*1000))
            else:
                init = generate_random_init(seed=7000 + int(time.time()*1000))
                no_improve_count = 0
                bh_sigma = 0.08
            
            remaining = deadline - 15 - time.time()
            if remaining < 3:
                break
            
            t_limit = min(4.0, remaining)
            refined = refine_local(init, time_limit=t_limit)
            
            if update_best(refined):
                no_improve_count = 0
                bh_sigma = max(0.02, bh_sigma * 0.9)
            else:
                no_improve_count += 1
                bh_sigma = min(0.15, bh_sigma * 1.1)
    
    # Final high-precision polish on the best candidate
    if best_bary is not None and time.time() < deadline - 5:
        # Progressive soft-min polish (beta 100 -> 300 -> 800)
        for beta in [100.0, 300.0, 800.0]:
            if time.time() > deadline - 3:
                break
            try:
                bounds = [(0, 1)] * (N * 2)
                result_soft = minimize(
                    soft_min_obj,
                    best_bary.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                if result_soft.success or result_soft.fun < -hard_min_obj(best_bary.flatten()):
                    best_bary = clamp_bary(result_soft.x.reshape(N, 2))
                    best_val = -hard_min_obj(best_bary.flatten())
            except Exception:
                pass
        
        if best_bary is not None and time.time() < deadline - 3:
            try:
                bounds = [(0, 1)] * (N * 2)
                result_slsqp = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-15}
                )
                update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
            except Exception:
                pass
        
        if best_bary is not None and time.time() < deadline - 2:
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-16, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
