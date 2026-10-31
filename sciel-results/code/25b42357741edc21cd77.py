import numpy as np
from scipy.optimize import minimize
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

def soft_min_obj(bary_flat, alpha=20):
    """Soft minimum approximation."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    # Soft min: -logsumexp(-areas * alpha) / alpha
    log_sum_exp = np.log(np.sum(np.exp(-areas * alpha))) / alpha
    return -log_sum_exp

def epigraph_obj(x):
    """
    Epigraph formulation: x = [bary (22 values), t (1 value)]
    Minimize -t subject to t <= area_i for all i.
    We maximize t, so minimize -t.
    Constraints: area_i - t >= 0 for all i.
    """
    bary = x[:N*2].reshape(N, 2)
    t = x[N*2]
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -t

def epigraph_constraints(x):
    """Constraints for epigraph: area_i - t >= 0"""
    bary = x[:N*2].reshape(N, 2)
    t = x[N*2]
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return areas - t

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration.
    """
    best_bary = None
    best_val = -np.inf
    
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
    
    # Stage 1: Smooth soft-min minimization
    try:
        result1 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
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
            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
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
            options={'maxiter': 600, 'ftol': 1e-14}
        )
        bary_current = clamp_bary(result3.x.reshape(N, 2))
    except Exception:
        pass
        
    return bary_current

def refine_epigraph(bary_init, time_limit=8.0):
    """
    Refine using epigraph formulation with SLSQP.
    This handles the non-smooth min by introducing an auxiliary variable.
    """
    start_time = time.time()
    
    # Initial t estimate: half the current min area
    areas = calc_areas_bary(bary_init)
    t_init = np.min(areas) * 0.99
    
    # Build initial x: [bary_flat, t]
    x_init = np.concatenate([bary_init.flatten(), [t_init]])
    
    # Bounds: bary coords in [0,1], t in [0, 1]
    bounds = [(0, 1)] * (N * 2) + [(0, 1)]
    
    # Define constraints
    cons = {'type': 'ineq', 'fun': epigraph_constraints}
    
    # SLSQP with epigraph
    try:
        result = minimize(
            epigraph_obj,
            x_init,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1500, 'ftol': 1e-14}
        )
        if result.success or result.fun < -t_init:
            bary_out = clamp_bary(result.x[:N*2].reshape(N, 2))
            return bary_out
    except Exception:
        pass
    
    if time.time() - start_time > time_limit:
        return bary_init
    
    # Try with different initial t
    for t_frac in [0.5, 0.8, 1.0, 1.1, 1.2]:
        if time.time() - start_time > time_limit:
            break
        t_init2 = np.min(areas) * t_frac
        x_init2 = np.concatenate([bary_init.flatten(), [t_init2]])
        try:
            result = minimize(
                epigraph_obj,
                x_init2,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 1000, 'ftol': 1e-13}
            )
            if result.success or result.fun < -t_init2:
                bary_out = clamp_bary(result.x[:N*2].reshape(N, 2))
                val = -hard_min_obj(bary_out.flatten())
                if val > -hard_min_obj(bary_init.flatten()):
                    return bary_out
        except Exception:
            pass
    
    return bary_init

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

    # 1. Try Hexagonal Init
    hex_init = generate_hexagonal_init()
    
    # Refine with local method first
    refined_hex = refine_local(hex_init, time_limit=5.0)
    update_best(refined_hex)
    
    # Then try epigraph refinement on the best so far
    if best_bary is not None and time.time() < deadline - 15:
        epigraph_refined = refine_epigraph(best_bary, time_limit=10.0)
        update_best(epigraph_refined)
    
    # 2. Multi-start: random + perturbed restarts with local refinement
    rng = np.random.default_rng(123)
    restart_count = 0
    while time.time() < deadline - 20:
        restart_count += 1
        if best_bary is not None and restart_count % 3 == 0:
            # Perturbation of best solution
            init = generate_perturbed_init(best_bary, sigma=0.03, seed=2000 + restart_count)
        elif best_bary is not None and restart_count % 3 == 1:
            # Larger perturbation
            init = generate_perturbed_init(best_bary, sigma=0.08, seed=2500 + restart_count)
        else:
            # Fresh random start
            init = generate_random_init(seed=3000 + restart_count)
        
        refined = refine_local(init, time_limit=3.0)
        update_best(refined)
        
        # Occasionally try epigraph on the current best
        if restart_count % 5 == 0 and best_bary is not None and time.time() < deadline - 25:
            epigraph_refined = refine_epigraph(best_bary, time_limit=5.0)
            update_best(epigraph_refined)

    # 3. Final high-precision polish
    if best_bary is not None and time.time() < deadline - 8:
        # Epigraph refinement for final push
        try:
            epigraph_final = refine_epigraph(best_bary, time_limit=6.0)
            update_best(epigraph_final)
        except Exception:
            pass
        
        # SLSQP polish
        if best_bary is not None and time.time() < deadline - 4:
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
                    options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass
        
        # One more SLSQP pass
        if best_bary is not None and time.time() < deadline - 0.5:
            try:
                bounds = [(0, 1)] * (N * 2)
                result_final = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-16}
                )
                update_best(clamp_bary(result_final.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
