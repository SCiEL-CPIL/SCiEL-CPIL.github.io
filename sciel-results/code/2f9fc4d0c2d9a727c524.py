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

def clamp_bary(bary, eps=1e-12):
    """Clamp barycentric coordinates to the valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    
    sums = u + v
    invalid = sums > 1.0
    if np.any(invalid):
        scale = 1.0 / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
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

def epigraph_obj(params):
    """
    Epigraph formulation: params = [bary (N*2), t]
    Minimize -t subject to soft penalty for areas < t.
    """
    bary_flat = params[:2*N]
    t = params[2*N]
    
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    violation = np.maximum(0, t - areas)
    penalty = np.sum(violation**2) * 5000
    return -t + penalty

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration.
    """
    best_bary = None
    best_val = -np.inf
    
    for scale in [0.12, 0.15, 0.18, 0.2, 0.22, 0.25, 0.28, 0.3, 0.35, 0.4]:
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
            rng = np.random.default_rng(42)
            for _ in range(10):
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

def generate_symmetric_init():
    """
    Generate a symmetric initial configuration.
    """
    bary = np.array([
        [0.05, 0.05],
        [0.05, 0.95],
        [0.95, 0.05],
        [0.05, 0.5],
        [0.5, 0.05],
        [0.5, 0.5],
        [0.25, 0.25],
        [0.75, 0.25],
        [0.5, 0.5],
        [0.25, 0.75],
        [0.75, 0.5]
    ])
    return clamp_bary(bary)

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
    
    # Stage 1: Coarse soft-min (low beta)
    try:
        result1 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(15.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        new_bary = clamp_bary(result1.x.reshape(N, 2))
        old_val = -hard_min_obj(bary_current.flatten())
        new_val = -hard_min_obj(new_bary.flatten())
        if new_val > old_val:
            bary_current = new_bary
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.25:
        return bary_current

    # Stage 2: Medium soft-min
    try:
        result2 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(40.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-9}
        )
        new_bary = clamp_bary(result2.x.reshape(N, 2))
        old_val = -hard_min_obj(bary_current.flatten())
        new_val = -hard_min_obj(new_bary.flatten())
        if new_val > old_val:
            bary_current = new_bary
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.5:
        return bary_current

    # Stage 3: Sharp soft-min (high beta, close to hard min)
    try:
        result3 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(150.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        new_bary = clamp_bary(result3.x.reshape(N, 2))
        old_val = -hard_min_obj(bary_current.flatten())
        new_val = -hard_min_obj(new_bary.flatten())
        if new_val > old_val:
            bary_current = new_bary
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.75:
        return bary_current

    # Stage 4: Nelder-Mead on hard min
    try:
        result4 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
        )
        new_bary = clamp_bary(result4.x.reshape(N, 2))
        old_val = -hard_min_obj(bary_current.flatten())
        new_val = -hard_min_obj(new_bary.flatten())
        if new_val > old_val:
            bary_current = new_bary
    except Exception:
        pass
        
    return bary_current

def refine_epigraph(bary_init, time_limit=5.0):
    """
    Refine using epigraph formulation with auxiliary t variable.
    """
    start_time = time.time()
    
    # Initialize t to current min area
    areas = calc_areas_bary(bary_init)
    t_init = np.min(areas)
    
    params_init = np.concatenate([bary_init.flatten(), [t_init]])
    
    # Bounds for params
    bounds = [(0, 1)] * (2*N) + [(0, 1)]
    
    # Stage 1: L-BFGS-B on epigraph
    try:
        result1 = minimize(
            epigraph_obj,
            params_init,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        new_params = result1.x
        new_bary = clamp_bary(new_params[:2*N].reshape(N, 2))
        old_val = -hard_min_obj(bary_init.flatten())
        new_val = -hard_min_obj(new_bary.flatten())
        if new_val > old_val:
            bary_init = new_bary
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.5:
        return bary_init
    
    # Stage 2: NM polish on hard min
    try:
        result2 = minimize(
            hard_min_obj,
            bary_init.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
        )
        new_bary = clamp_bary(result2.x.reshape(N, 2))
        old_val = -hard_min_obj(bary_init.flatten())
        new_val = -hard_min_obj(new_bary.flatten())
        if new_val > old_val:
            bary_init = new_bary
    except Exception:
        pass
        
    return bary_init

def coordinate_ascent(bary_init, time_limit=5.0):
    """
    Coordinate ascent: optimize each point one at a time.
    This is a different structural approach from the gradient-based methods.
    """
    start_time = time.time()
    bary_current = bary_init.copy()
    bounds = [(0, 1)] * 2
    
    for iteration in range(50):
        if time.time() - start_time > time_limit:
            break
            
        improved = False
        for i in range(N):
            if time.time() - start_time > time_limit:
                    break
                    
            # Fix all other points, optimize point i
            fixed_bary = bary_current.copy()
            
            def obj_point_i(x):
                fixed_bary[i] = x
                fixed_bary = clamp_bary(fixed_bary)
                areas = calc_areas_bary(fixed_bary)
                return -np.min(areas)
            
            try:
                result = minimize(
                    obj_point_i,
                    bary_current[i],
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-14, 'adaptive': True}
                )
                new_point = clamp_bary(np.array([result.x[0], result.x[1]])).flatten()
                old_val = -hard_min_obj(bary_current.flatten())
                bary_current[i] = new_point
                new_val = -hard_min_obj(bary_current.flatten())
                if new_val > old_val + 1e-12:
                    improved = True
            except Exception:
                pass
        
        if not improved:
            break
    
    return bary_current

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

    # 1. Try Hexagonal Init with local refinement
    hex_init = generate_hexagonal_init()
    refined_hex = refine_local(hex_init, time_limit=10.0)
    update_best(refined_hex)
    
    # 2. Try coordinate ascent on hex init
    if time.time() < deadline - 100:
        ca_hex = coordinate_ascent(hex_init, time_limit=8.0)
        update_best(ca_hex)
    
    # 3. Try symmetric init
    sym_init = generate_symmetric_init()
    refined_sym = refine_local(sym_init, time_limit=10.0)
    update_best(refined_sym)
    
    # 4. Try coordinate ascent on symmetric init
    if time.time() < deadline - 90:
        ca_sym = coordinate_ascent(sym_init, time_limit=8.0)
        update_best(ca_sym)
    
    # 5. Global Search with Differential Evolution
    if time.time() < deadline - 60:
        try:
            def de_obj(x):
                bary = x.reshape(N, 2)
                bary = clamp_bary(bary)
                areas = calc_areas_bary(bary)
                return -np.min(areas)
            
            bounds = [(0, 1)] * (N * 2)
            result_de = differential_evolution(
                de_obj,
                bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                polish=False
            )
            
            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined_de = refine_local(de_bary, time_limit=10.0)
            update_best(refined_de)
            
            if time.time() < deadline - 45:
                ca_de = coordinate_ascent(de_bary, time_limit=6.0)
                update_best(ca_de)
        except Exception:
            pass
    
    # 6. Multi-start: random + perturbed restarts
    rng = np.random.default_rng(123)
    restart_count = 0
    while time.time() < deadline - 20:
        restart_count += 1
        if best_bary is not None and restart_count % 2 == 0:
            init = generate_perturbed_init(best_bary, sigma=0.03, seed=2000 + restart_count)
        else:
            init = generate_random_init(seed=3000 + restart_count)
        
        # Alternate between refinement strategies
        if restart_count % 3 == 0:
            refined = coordinate_ascent(init, time_limit=4.0)
        elif restart_count % 3 == 1:
            refined = refine_local(init, time_limit=4.0)
        else:
            refined = refine_epigraph(init, time_limit=4.0)
        update_best(refined)

    # 7. Final high-precision polish
    if best_bary is not None and time.time() < deadline - 10:
        try:
            bounds = [(0, 1)] * (N * 2)
            result_slsqp = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1500, 'ftol': 1e-15}
            )
            update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
        except Exception:
            pass
        
        if best_bary is not None and time.time() < deadline - 6:
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass
        
        # One more SLSQP pass
        if best_bary is not None and time.time() < deadline - 3:
            try:
                bounds = [(0, 1)] * (N * 2)
                result_final = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 3000, 'ftol': 1e-16}
                )
                update_best(clamp_bary(result_final.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
