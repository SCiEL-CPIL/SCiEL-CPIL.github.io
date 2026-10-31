import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 145.0  # Safety margin for 150s limit

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Fixed Delaunay-like topology for 11 points (17 triangles)
FIXED_TRI = [
    (0, 1, 3), (0, 3, 4), (0, 4, 2),
    (1, 3, 5), (1, 5, 6),
    (2, 4, 7), (2, 7, 6),
    (3, 5, 8), (3, 8, 4),
    (4, 8, 9), (4, 9, 7),
    (5, 6, 10), (5, 10, 8),
    (6, 7, 10), (6, 10, 9),
    (7, 9, 10), (8, 9, 10),
]
FIXED_TRI_ARR = np.array(FIXED_TRI)

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

def fixed_tri_areas(bary):
    """Compute areas of only the fixed-topology triangles."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    ui = u[FIXED_TRI_ARR[:, 0]]
    vi = v[FIXED_TRI_ARR[:, 0]]
    wi = w[FIXED_TRI_ARR[:, 0]]
    uj = u[FIXED_TRI_ARR[:, 1]]
    vj = v[FIXED_TRI_ARR[:, 1]]
    wj = w[FIXED_TRI_ARR[:, 1]]
    uk = u[FIXED_TRI_ARR[:, 2]]
    vk = v[FIXED_TRI_ARR[:, 2]]
    wk = w[FIXED_TRI_ARR[:, 2]]
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    return np.abs(det)

def fixed_min_obj(bary_flat):
    """Maximize min area over FIXED topology triangles only."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = fixed_tri_areas(bary)
    return -np.min(areas)

def fixed_soft_obj(bary_flat, beta=30.0):
    """Smooth min over fixed topology triangles."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = fixed_tri_areas(bary)
    if np.any(areas < 1e-15):
        return 1e10
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        return -np.min(areas)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    return -soft_min

def epigraph_fixed_obj(params, mu=200.0):
    """Epigraph on fixed topology triangles."""
    bary_flat = params[:N*2]
    t = params[N*2]
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = fixed_tri_areas(bary)
    violations = np.maximum(0, t - areas)
    penalty = np.sum(violations**2)
    return -t + mu * penalty

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

def epigraph_penalty_obj(params, mu=100.0):
    """
    Epigraph formulation with penalty.
    params = [bary_flat (22), t]
    Minimize -t + mu * sum(max(0, t - area_i)^2)
    """
    bary_flat = params[:N*2]
    t = params[N*2]
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Violations: t - area_i where t > area_i
    violations = np.maximum(0, t - areas)
    penalty = np.sum(violations**2)
    
    return -t + mu * penalty

def generate_structured_init():
    """
    Generate initial configuration matching the fixed topology structure.
    Point layout: 0=vertex0, 1=vertex1, 2=vertex2, 3-4=edge mids, 5-7=mid ring, 8-9=inner, 10=center
    """
    bary = np.zeros((N, 2))
    cx, cy = 1/3, 1/3
    
    bary[0] = [0.88, 0.06]
    bary[1] = [0.06, 0.88]
    bary[2] = [0.06, 0.06]
    bary[3] = [0.42, 0.08]
    bary[4] = [0.42, 0.42]
    
    r_mid = 0.22
    for i in range(3):
        angle = 2 * np.pi * i / 3 + np.pi / 6
        bary[5 + i] = [cx + r_mid * np.cos(angle), cy + r_mid * np.sin(angle)]
    
    r_in = 0.12
    for i in range(2):
        angle = 2 * np.pi * i / 2 + np.pi / 4
        bary[8 + i] = [cx + r_in * np.cos(angle), cy + r_in * np.sin(angle)]
    
    bary[10] = [cx, cy]
    return clamp_bary(bary)

def generate_hex_init():
    """Hexagonal lattice inspired init matching fixed topology."""
    bary = np.zeros((N, 2))
    cx, cy = 1/3, 1/3
    bary[0] = [0.9, 0.05]
    bary[1] = [0.05, 0.9]
    bary[2] = [0.05, 0.05]
    bary[3] = [0.45, 0.05]
    bary[4] = [0.45, 0.45]
    for i in range(3):
        angle = 2 * np.pi * i / 3
        bary[5 + i] = [cx + 0.2 * np.cos(angle), cy + 0.2 * np.sin(angle)]
    for i in range(2):
        angle = 2 * np.pi * i / 2 + np.pi / 4
        bary[8 + i] = [cx + 0.1 * np.cos(angle), cy + 0.1 * np.sin(angle)]
    bary[10] = [cx, cy]
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

def refine_fixed(bary_init, time_limit=6.0):
    """
    Multi-stage refinement focusing on fixed topology first, then full polish.
    Stage 1: Soft min on fixed triangles (L-BFGS-B)
    Stage 2: Epigraph on fixed triangles (SLSQP)
    Stage 3: Full min-area Nelder-Mead polish
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    bary_cur = bary_init.copy()
    
    # Stage 1: Soft min on fixed topology
    try:
        r1 = minimize(fixed_soft_obj, bary_cur.flatten(), args=(30.0,),
                      method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
        if r1.fun < -hard_min_obj(bary_cur.flatten()):
            bary_cur = clamp_bary(r1.x.reshape(N, 2))
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.35:
        return bary_cur
    
    # Stage 2: Epigraph on fixed topology
    t_init = -fixed_min_obj(bary_cur.flatten())
    params = np.concatenate([bary_cur.flatten(), [t_init]])
    bounds_ep = bounds + [(0, 1)]
    try:
        r2 = minimize(epigraph_fixed_obj, params, args=(200.0,),
                      method='SLSQP', bounds=bounds_ep,
                      options={'maxiter': 800, 'ftol': 1e-14})
        new_bary = clamp_bary(r2.x[:N*2].reshape(N, 2))
        if -hard_min_obj(new_bary.flatten()) > -hard_min_obj(bary_cur.flatten()):
            bary_cur = new_bary
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.7:
        return bary_cur
    
    # Stage 3: Full min-area Nelder-Mead polish
    try:
        r3 = minimize(hard_min_obj, bary_cur.flatten(),
                      method='Nelder-Mead',
                      options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True})
        bary_cur = clamp_bary(r3.x.reshape(N, 2))
    except Exception:
        pass
    
    return bary_cur

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

    # 1. Structured init with fixed-topology refinement
    struct_init = generate_structured_init()
    refined = refine_fixed(struct_init, time_limit=10.0)
    update_best(refined)
    
    # 2. Hex init
    hex_init = generate_hex_init()
    refined = refine_fixed(hex_init, time_limit=10.0)
    update_best(refined)
    
    # 3. Differential evolution on full problem
    if time.time() < deadline - 40:
        try:
            def de_obj(x):
                bary = x.reshape(N, 2)
                bary = clamp_bary(bary)
                return -np.min(calc_areas_bary(bary))
            
            bounds = [(0, 1)] * (N * 2)
            result_de = differential_evolution(
                de_obj, bounds, seed=42,
                maxiter=50, popsize=18, tol=1e-6, polish=False)
            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined = refine_fixed(de_bary, time_limit=10.0)
            update_best(refined)
        except Exception:
            pass
    
    # 4. Multi-start with perturbation of best
    restart_count = 0
    while time.time() < deadline - 15:
        restart_count += 1
        if best_bary is not None and restart_count % 2 == 0:
            init = generate_perturbed_init(best_bary, sigma=0.03, seed=5000 + restart_count)
        else:
            init = generate_random_init(seed=6000 + restart_count)
        
        remaining = deadline - 15 - time.time()
        if remaining < 4:
            break
        t_lim = min(6.0, remaining)
        refined = refine_fixed(init, time_limit=t_lim)
        update_best(refined)

    # Final high-precision polish on the best candidate
    if best_bary is not None and time.time() < deadline - 5:
        try:
            # One more epigraph polish
            t_init = -hard_min_obj(best_bary.flatten())
            params_init = np.concatenate([best_bary.flatten(), [t_init]])
            bounds_ep = [(0, 1)] * (N * 2) + [(0, 1)]
            
            result_slsqp = minimize(
                epigraph_penalty_obj,
                params_init,
                args=(1000.0,), # Higher penalty for final polish
                method='SLSQP',
                bounds=bounds_ep,
                options={'maxiter': 1000, 'ftol': 1e-15}
            )
            if result_slsqp.success:
                new_bary = clamp_bary(result_slsqp.x[:N*2].reshape(N, 2))
                update_best(new_bary)
        except Exception:
            pass
        
        if best_bary is not None and time.time() < deadline - 2:
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-13, 'fatol': 1e-16, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
