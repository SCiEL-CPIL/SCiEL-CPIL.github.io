import numpy as np
from scipy.optimize import minimize, differential_evolution
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

def soft_min_obj(bary_flat, beta=50.0, var_penalty=0.0):
    """
    Smooth surrogate for the hard min via log-sum-exp.
    Includes an optional variance penalty to encourage 'Voronoi-like' uniform distribution.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    if np.any(areas < 1e-15):
        return 1e10
    
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    
    if max_scaled > 700:
        # Fallback to hard min if numerical instability occurs
        val = -np.min(areas)
        if var_penalty > 0:
            val += var_penalty * np.var(areas)
        return val
        
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    
    obj_val = -soft_min
    if var_penalty > 0:
        # Penalize variance to push towards uniform area distribution
        obj_val += var_penalty * np.var(areas)
        
    return obj_val

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

def soft_min_obj(bary_flat, beta=50.0, var_penalty=0.0):
    """
    Smooth surrogate for the hard min via log-sum-exp.
    Includes an optional variance penalty to encourage 'Voronoi-like' uniform distribution.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    if np.any(areas < 1e-15):
        return 1e10
    
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    
    if max_scaled > 700:
        # Fallback to hard min if numerical instability occurs
        val = -np.min(areas)
        if var_penalty > 0:
            val += var_penalty * np.var(areas)
        return val
        
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    
    obj_val = -soft_min
    if var_penalty > 0:
        # Penalize variance to push towards uniform area distribution
        obj_val += var_penalty * np.var(areas)
        
    return obj_val

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

def generate_centroid_init():
    """
    Generate a symmetric configuration with one point at centroid and others on concentric rings.
    """
    bary = np.zeros((N, 2))
    # Centroid
    bary[0] = [1/3, 1/3]
    
    # Ring 1: 6 points around centroid
    r1 = 0.15
    for i in range(6):
        angle = 2 * np.pi * i / 6
        u = 1/3 + r1 * np.cos(angle)
        v = 1/3 + r1 * np.sin(angle)
        bary[i+1] = [u, v]
    
    # Remaining 4 points: place at midpoints of edges
    bary[7] = [0.5, 0.0]  # Midpoint of bottom edge
    bary[8] = [0.0, 0.5]  # Midpoint of left edge  
    bary[9] = [0.5, 0.5]  # Midpoint of right edge (in barycentric)
    bary[10] = [1/3, 1/3 + 0.3]  # Top
    
    return clamp_bary(bary)

def generate_corner_init():
    """
    Generate configuration with points near corners and distributed inside.
    """
    bary = np.zeros((N, 2))
    # Near corners
    bary[0] = [0.05, 0.05]
    bary[1] = [0.95, 0.05]
    bary[2] = [0.05, 0.95]
    
    # Centroid
    bary[3] = [1/3, 1/3]
    
    # Mid-edges
    bary[4] = [0.5, 0.05]
    bary[5] = [0.05, 0.5]
    bary[6] = [0.5, 0.5]
    
    # Inner ring
    r = 0.2
    for i in range(4):
        angle = 2 * np.pi * i / 4 + np.pi/4
        u = 1/3 + r * np.cos(angle)
        v = 1/3 + r * np.sin(angle)
        bary[7+i] = [u, v]
    
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
    Refine a configuration using multiple optimization stages with graduated soft-min.
    Stages:
    1. Coarse soft-min with variance penalty (explore uniform landscape)
    2. Medium soft-min with variance penalty
    3. Sharp soft-min (reduce variance penalty, focus on min area)
    4. Hard min refinement with Nelder-Mead
    5. Final high-precision Nelder-Mead polish
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    current_val = -hard_min_obj(bary_current.flatten())
    
    # Stage 1: Coarse soft-min with variance penalty to encourage spread
    try:
        result1 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(20.0, 0.01),  # beta=20, var_penalty=0.01
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        new_val = -hard_min_obj(result1.x.reshape(N, 2).flatten())
        if new_val > current_val:
            bary_current = clamp_bary(result1.x.reshape(N, 2))
            current_val = new_val
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.25:
        return bary_current

    # Stage 2: Medium soft-min with reduced variance penalty
    try:
        result2 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(50.0, 0.005), # beta=50, var_penalty=0.005
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-9}
        )
        new_val = -hard_min_obj(result2.x.reshape(N, 2).flatten())
        if new_val > current_val:
            bary_current = clamp_bary(result2.x.reshape(N, 2))
            current_val = new_val
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.45:
        return bary_current

    # Stage 3: Sharp soft-min (no variance penalty, focus purely on min area)
    try:
        result3 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(200.0, 0.0),  # beta=200, var_penalty=0
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        new_val = -hard_min_obj(result3.x.reshape(N, 2).flatten())
        if new_val > current_val:
            bary_current = clamp_bary(result3.x.reshape(N, 2))
            current_val = new_val
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.65:
        return bary_current

    # Stage 4: Hard min refinement with Nelder-Mead
    try:
        result4 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1500, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
        )
        bary_current = clamp_bary(result4.x.reshape(N, 2))
        current_val = -hard_min_obj(bary_current.flatten())
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.85:
        return bary_current

    # Stage 5: Final high-precision Nelder-Mead polish
    try:
        result5 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-15, 'fatol': 1e-17, 'adaptive': True}
        )
        bary_current = clamp_bary(result5.x.reshape(N, 2))
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

    # 1. Try Hexagonal Init
    hex_init = generate_hexagonal_init()
    refined_hex = refine_local(hex_init, time_limit=8.0)
    update_best(refined_hex)

    # 2. Try Centroid Init
    if time.time() < deadline - 30:
        centroid_init = generate_centroid_init()
        refined_centroid = refine_local(centroid_init, time_limit=5.0)
        update_best(refined_centroid)

    # 3. Try Corner Init
    if time.time() < deadline - 25:
        corner_init = generate_corner_init()
        refined_corner = refine_local(corner_init, time_limit=5.0)
        update_best(refined_corner)

    # 4. Global Search with Differential Evolution
    if time.time() < deadline - 30:
        try:
            def de_obj(x):
                bary = x.reshape(N, 2)
                bary = clamp_bary(bary)
                areas = calc_areas_bary(bary)
                # DE objective: maximize min area, slight penalty for variance to guide search
                return -np.min(areas) + 0.001 * np.var(areas)

            bounds = [(0, 1)] * (N * 2)
            result_de = differential_evolution(
                de_obj,
                bounds,
                seed=42,
                maxiter=30,
                popsize=12,
                tol=1e-6,
                polish=False
            )

            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined_de = refine_local(de_bary, time_limit=8.0)
            update_best(refined_de)
        except Exception:
            pass

    # 5. Multi-start: random + perturbed restarts with local refinement
    rng = np.random.default_rng(123)
    restart_count = 0
    while time.time() < deadline - 15:
        restart_count += 1
        if best_bary is not None and restart_count % 2 == 0:
            # Perturbation of best solution
            init = generate_perturbed_init(best_bary, sigma=0.04, seed=2000 + restart_count)
        else:
            # Fresh random start
            init = generate_random_init(seed=3000 + restart_count)
        
        refined = refine_local(init, time_limit=3.0)
        update_best(refined)

    # 4. Final high-precision polish with graduated approach
    if best_bary is not None and time.time() < deadline - 8:
        # Graduated soft-min polish: increase beta progressively, no variance penalty in final stages
        for beta in [100.0, 300.0, 800.0]:
            if time.time() > deadline - 3:
                break
            try:
                bounds = [(0, 1)] * (N * 2)
                result_soft = minimize(
                    soft_min_obj,
                    best_bary.flatten(),
                    args=(beta, 0.0),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-11}
                )
                update_best(clamp_bary(result_soft.x.reshape(N, 2)))
            except Exception:
                pass
        
        if best_bary is not None and time.time() < deadline - 3:
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass
        
        # Ultra-precise final NM pass
        if best_bary is not None and time.time() < deadline - 1:
            try:
                result_final = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-16, 'fatol': 1e-18, 'adaptive': True}
                )
                update_best(clamp_bary(result_final.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
