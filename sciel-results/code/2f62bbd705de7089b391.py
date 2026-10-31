import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import combinations
import time

N = 11
TIME_LIMIT = 148.0  # Safety margin for 150s limit

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
    Minimizing this pushes ALL areas up, with emphasis on the smallest.
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

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration with more scales and patterns.
    """
    best_bary = None
    best_val = -np.inf
    
    # Try multiple scales and offsets for better coverage
    for scale in [0.12, 0.15, 0.18, 0.2, 0.22, 0.25, 0.28, 0.3, 0.35, 0.4]:
        for offset in [0.0, 0.5 * scale]:
            pts = []
            max_k = int(1.0 / scale) + 2
            for k in range(max_k + 1):
                w_val = k * scale + offset
                if w_val > 1.0 - 1e-9:
                    break
                if w_val < -1e-9:
                    continue
                w_val = max(0.0, w_val)
                num_pts_in_row = int((1 - w_val) / scale) + 2
                for j in range(num_pts_in_row):
                    u_val = j * scale + offset
                    if u_val > 1.0 - 1e-9:
                        break
                    if u_val < -1e-9:
                        continue
                    u_val = max(0.0, u_val)
                    v_val = (1 - w_val) - u_val
                    if v_val < -1e-9:
                        continue
                    v_val = max(0.0, v_val)
                    if u_val + v_val <= 1 + 1e-9:
                        pts.append([u_val, v_val])
            
            if len(pts) >= N:
                # Try different subsets with different random seeds
                rng = np.random.default_rng(42 + int(scale * 1000) + int(offset * 1000))
                for trial in range(8):
                    if len(pts) == N:
                        bary = np.array(pts)
                    else:
                        idx = rng.choice(len(pts), N, replace=False)
                        bary = np.array(pts)[idx]
                    val = -hard_min_obj(bary.flatten())
                    if val > best_val:
                        best_val = val
                        best_bary = bary.copy()
    
    # Also try a simple grid pattern
    for grid_n in [3, 4, 5]:
        pts = []
        for i in range(grid_n):
            for j in range(grid_n - i):
                u_val = i / (grid_n - 1) if grid_n > 1 else 0
                v_val = j / (grid_n - 1) if grid_n > 1 else 0
                w_val = 1 - u_val - v_val
                if w_val >= -1e-9:
                    pts.append([u_val, v_val])
        
        if len(pts) >= N:
            rng = np.random.default_rng(777)
            for trial in range(5):
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
    Refine a configuration using multiple optimization stages with graduated soft-min.
    6-stage approach: coarse -> medium -> sharp -> very_sharp -> NM -> NM polish
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    cur_val = -hard_min_obj(bary_current.flatten())
    
    # Stage 1: Coarse soft-min (low beta, more iterations)
    try:
        result1 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(15.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        new_val = -hard_min_obj(clamp_bary(result1.x.reshape(N, 2)).flatten())
        if new_val > cur_val:
            bary_current = clamp_bary(result1.x.reshape(N, 2))
            cur_val = new_val
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.2:
        return bary_current

    # Stage 2: Medium soft-min (increasing beta)
    try:
        result2 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(40.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-9}
        )
        new_val = -hard_min_obj(clamp_bary(result2.x.reshape(N, 2)).flatten())
        if new_val > cur_val:
            bary_current = clamp_bary(result2.x.reshape(N, 2))
            cur_val = new_val
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.4:
        return bary_current

    # Stage 3: Sharp soft-min (high beta, close to hard min)
    try:
        result3 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(100.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        new_val = -hard_min_obj(clamp_bary(result3.x.reshape(N, 2)).flatten())
        if new_val > cur_val:
            bary_current = clamp_bary(result3.x.reshape(N, 2))
            cur_val = new_val
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.55:
        return bary_current

    # Stage 4: Very sharp soft-min (very high beta)
    try:
        result4 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(300.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-11}
        )
        new_val = -hard_min_obj(clamp_bary(result4.x.reshape(N, 2)).flatten())
        if new_val > cur_val:
            bary_current = clamp_bary(result4.x.reshape(N, 2))
            cur_val = new_val
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.7:
        return bary_current

    # Stage 5: Hard min refinement with Nelder-Mead
    try:
        result5 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
        )
        bary_current = clamp_bary(result5.x.reshape(N, 2))
        cur_val = -hard_min_obj(bary_current.flatten())
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.85:
        return bary_current

    # Stage 6: Final high-precision Nelder-Mead polish
    try:
        result6 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1500, 'xatol': 1e-15, 'fatol': 1e-17, 'adaptive': True}
        )
        bary_current = clamp_bary(result6.x.reshape(N, 2))
    except Exception:
        pass
        
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

    # 1. Try Hexagonal Init with standard refinement
    hex_init = generate_hexagonal_init()
    refined_hex = refine_local(hex_init, time_limit=8.0)
    update_best(refined_hex)

    # 2. Global Search with Differential Evolution (improved)
    if time.time() < deadline - 40:
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
                tol=1e-7,
                polish=False,
                mutation=(0.5, 1.5),
                recombination=0.7
            )

            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined_de = refine_local(de_bary, time_limit=10.0)
            update_best(refined_de)
        except Exception:
            pass

    # 3. Multi-start: random + perturbed restarts with local refinement
    rng = np.random.default_rng(123)
    restart_count = 0
    while time.time() < deadline - 15:
        restart_count += 1
        if best_bary is not None and restart_count % 3 == 0:
            # Perturbation of best solution with varying sigma
            sigma = 0.02 + 0.03 * (restart_count % 5) / 5.0
            init = generate_perturbed_init(best_bary, sigma=sigma, seed=2000 + restart_count)
        elif best_bary is not None and restart_count % 3 == 1:
            # Moderate perturbation
            init = generate_perturbed_init(best_bary, sigma=0.06, seed=2500 + restart_count)
        else:
            # Fresh random start
            init = generate_random_init(seed=3000 + restart_count)
        
        refined = refine_local(init, time_limit=4.0)
        update_best(refined)

    # 4. Final high-precision polish
    if best_bary is not None and time.time() < deadline - 8:
        try:
            bounds = [(0, 1)] * (N * 2)
            result_slsqp = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1500, 'ftol': 1e-16}
            )
            update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
        except Exception:
            pass
        
        if best_bary is not None and time.time() < deadline - 5:
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 4000, 'xatol': 1e-15, 'fatol': 1e-18, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass
        
        # One more SLSQP pass with tighter tolerance
        if best_bary is not None and time.time() < deadline - 1:
            try:
                bounds = [(0, 1)] * (N * 2)
                result_final = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 3000, 'ftol': 1e-17}
                )
                update_best(clamp_bary(result_final.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
