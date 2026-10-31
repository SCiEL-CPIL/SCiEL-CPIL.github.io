import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 145.0

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)

H = np.sqrt(3) / 2

def bary_to_cart(bary):
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def calc_areas_bary(bary):
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
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def soft_min_obj(bary_flat, beta=100.0):
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

def params_to_bary(params):
    """
    Convert structured parameters to barycentric coordinates.
    
    Parameter layout (15 params):
    0: r_vertex (distance of vertex points from actual vertices, 0-0.3)
    1: r_edge (distance of edge points from midpoints, 0-0.3)  
    2: r_ring1 (radius of first interior ring, 0-0.5)
    3: theta_ring1 (angle offset of first ring, 0-2pi)
    4: r_ring2 (radius of second interior ring, 0-0.5)
    5: theta_ring2 (angle offset of second ring, 0-2pi)
    6: center_u (u-coordinate of center point)
    7: center_v (v-coordinate of center point)
    8-14: 7 additional free parameters for fine-tuning (small perturbations)
    """
    r_v = params[0]
    r_e = params[1]
    r_r1 = params[2]
    th_r1 = params[3]
    r_r2 = params[4]
    th_r2 = params[5]
    c_u = params[6]
    c_v = params[7]
    
    # Vertex layer: 3 points near vertices
    # V0=(0,0), V1=(1,0), V2=(0,1) in barycentric
    # Move each vertex inward by r_v along the angle bisector
    pts = []
    
    # Point near V0: move toward centroid (1/3, 1/3)
    pts.append([r_v / 2, r_v / 2])
    
    # Point near V1: move toward centroid
    pts.append([1 - r_v, r_v / 2])
    
    # Point near V2: move toward centroid  
    pts.append([r_v / 2, 1 - r_v])
    
    # Edge midpoint layer: 3 points near edge midpoints
    # Mid V0-V1 = (0.5, 0), Mid V1-V2 = (0.5, 0.5), Mid V0-V2 = (0, 0.5)
    # Move each inward by r_e
    
    # Near mid V0-V1: move up
    pts.append([0.5 - r_e / 2, r_e])
    
    # Near mid V1-V2: move toward centroid
    pts.append([0.5 - r_e / 2, 0.5 - r_e / 2])
    
    # Near mid V0-V2: move right
    pts.append([r_e, 0.5 - r_e / 2])
    
    # Interior ring 1: 3 points on a "circle" of radius r_r1 around center
    # In barycentric, use a hexagonal approximation
    cx, cy = 1/3, 1/3
    for k in range(3):
        angle = th_r1 + k * 2 * np.pi / 3
        # Approximate circular motion in barycentric space
        du = r_r1 * np.cos(angle) * 0.866
        dv = r_r1 * np.sin(angle)
        u = cx + du
        v = cy + dv
        # Clamp to valid region
        if u < 0: u = 0.01
        if v < 0: v = 0.01
        if u + v > 1: 
            scale = 0.99 / (u + v)
            u *= scale
            v *= scale
        pts.append([u, v])
    
    # Interior ring 2: 3 points on a smaller ring
    for k in range(3):
        angle = th_r2 + k * 2 * np.pi / 3
        du = r_r2 * np.cos(angle) * 0.866
        dv = r_r2 * np.sin(angle)
        u = cx + du
        v = cy + dv
        if u < 0: u = 0.01
        if v < 0: v = 0.01
        if u + v > 1:
            scale = 0.99 / (u + v)
            u *= scale
            v *= scale
        pts.append([u, v])
    
    # Center point
    pts.append([c_u, c_v])
    
    bary = np.array(pts[:N])
    
    # Apply small perturbations from free parameters
    if len(params) >= 15:
        for i in range(min(7, N)):
            bary[i, 0] += params[8 + i] * 0.02
            bary[i, 1] += params[8 + i] * 0.01
    
    return clamp_bary(bary)

def structured_obj(params):
    """Objective for structured parameter space."""
    bary = params_to_bary(params)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def generate_structured_init(seed=None):
    """Generate a good structured initial configuration."""
    rng = np.random.default_rng(seed)
    
    # Try several reasonable parameter combinations
    best_params = None
    best_val = -np.inf
    
    # Vertex points slightly inside
    r_v_options = [0.05, 0.1, 0.15]
    # Edge points slightly inside  
    r_e_options = [0.05, 0.1, 0.15]
    # First ring radius
    r_r1_options = [0.15, 0.2, 0.25]
    # Second ring radius
    r_r2_options = [0.08, 0.12, 0.16]
    # Center near centroid
    c_options = [(1/3, 1/3), (0.35, 0.35), (0.32, 0.34)]
    
    for rv in r_v_options:
        for re in r_e_options:
            for rr1 in r_r1_options:
                for rr2 in r_r2_options:
                    for cu, cv in c_options:
                        for th1 in [0, np.pi/6, np.pi/3]:
                            for th2 in [0, np.pi/6, np.pi/3]:
                                params = np.array([rv, re, rr1, th1, rr2, th2, cu, cv] + 
                                                  [0.0] * 7)
                                val = -structured_obj(params)
                                if val > best_val:
                                    best_val = val
                                    best_params = params.copy()
    
    if best_params is None:
        best_params = np.array([0.1, 0.1, 0.2, 0, 0.12, 0, 1/3, 1/3] + [0.0] * 7)
    
    return best_params

def refine_local(bary_init, time_limit=5.0):
    """Refine a configuration using multiple optimization stages."""
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    bary_current = bary_init.copy()
    
    # Stage 1: Soft-min with L-BFGS-B
    for beta in [50.0, 100.0, 200.0]:
        if time.time() - start_time > time_limit * 0.3:
            break
        try:
            result1 = minimize(
                soft_min_obj,
                bary_current.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if result1.success or result1.fun < -hard_min_obj(bary_current.flatten()):
                bary_current = clamp_bary(result1.x.reshape(N, 2))
        except Exception:
            pass
    
    if time.time() - start_time > time_limit * 0.5:
        return bary_current
    
    # Stage 2: Nelder-Mead on hard min
    try:
        result2 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
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
            options={'maxiter': 1000, 'ftol': 1e-16}
        )
        bary_current = clamp_bary(result3.x.reshape(N, 2))
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
    
    # Phase 1: Structured search in reduced parameter space
    if time.time() < deadline - 30:
        init_params = generate_structured_init(seed=42)
        
        # Refine in structured space
        structured_bounds = [
            (0.0, 0.3),   # r_vertex
            (0.0, 0.3),   # r_edge
            (0.0, 0.5),   # r_ring1
            (0, 2*np.pi), # theta_ring1
            (0.0, 0.5),   # r_ring2
            (0, 2*np.pi), # theta_ring2
            (0.1, 0.5),   # center_u
            (0.1, 0.5),   # center_v
        ] + [(-1.0, 1.0)] * 7
        
        try:
            result_struct = minimize(
                structured_obj,
                init_params,
                method='L-BFGS-B',
                bounds=structured_bounds,
                options={'maxiter': 500, 'ftol': 1e-14}
            )
            if result_struct.success:
                bary_from_struct = params_to_bary(result_struct.x)
                update_best(bary_from_struct)
        except Exception:
            pass
        
        # Try differential evolution in structured space
        if time.time() < deadline - 20:
            try:
                de_result = differential_evolution(
                    structured_obj,
                    structured_bounds,
                    seed=42,
                    maxiter=30,
                    popsize=12,
                    tol=1e-8,
                    polish=False
                )
                bary_from_de = params_to_bary(de_result.x)
                update_best(bary_from_de)
            except Exception:
                pass
    
    # Phase 2: Local refinement of best structured solution
    if best_bary is not None and time.time() < deadline - 15:
        refined = refine_local(best_bary, time_limit=10.0)
        update_best(refined)
    
    # Phase 3: Multi-start with random and perturbed initializations
    rng = np.random.default_rng(123)
    restart_count = 0
    while time.time() < deadline - 10:
        restart_count += 1
        if best_bary is not None and restart_count % 2 == 0:
            sigma = 0.03 + 0.02 * (restart_count % 3)
            noise = rng.normal(0, sigma, (N, 2))
            init = clamp_bary(best_bary + noise)
        else:
            u = rng.random(N)
            v = rng.random(N) * (1 - u)
            init = np.column_stack([u, v])
        
        refined = refine_local(init, time_limit=4.0)
        update_best(refined)
    
    # Phase 4: Final high-precision polish
    if best_bary is not None and time.time() < deadline - 5:
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
        
        if best_bary is not None and time.time() < deadline - 2:
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-15, 'fatol': 1e-19, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass
    
    cart = bary_to_cart(best_bary)
    return cart
