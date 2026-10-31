import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
from itertools import combinations

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

def generate_structured_inits():
    """
    Generate structured initial configurations based on known good patterns.
    """
    inits = []
    
    # Pattern 1: Hexagonal close packing subset
    center = np.array([1.0/3, 1.0/3])
    
    for r1 in [0.12, 0.15, 0.18, 0.22]:
        for r2 in [0.35, 0.45, 0.55]:
            pts = [center.copy()]
            
            for k in range(6):
                theta = np.pi/3 * k + np.pi/6
                dx = r1 * np.cos(theta)
                dy = r1 * np.sin(theta)
                du = -dx / 2 + dy / H
                dv = dx / 2 + dy / H
                pt = center + np.array([du, dv])
                if pt[0] > 0 and pt[1] > 0 and pt[0] + pt[1] < 1:
                    pts.append(pt.copy())
            
            if len(pts) < N:
                for k in range(6):
                    if len(pts) >= N:
                        break
                    theta = np.pi/3 * k
                    dx = r2 * np.cos(theta)
                    dy = r2 * np.sin(theta)
                    du = -dx / 2 + dy / H
                    dv = dx / 2 + dy / H
                    pt = center + np.array([du, dv])
                    if pt[0] > 0.01 and pt[1] > 0.01 and pt[0] + pt[1] < 0.98:
                        pts.append(pt.copy())
            
            if len(pts) == N:
                inits.append(np.array(pts))
    
    # Pattern 2: 3 near vertices, 3 near midpoints, 1 center, 4 distributed
    for vertex_offset in [0.05, 0.08, 0.12]:
        pts = []
        pts.append(np.array([vertex_offset, vertex_offset]))
        pts.append(np.array([1 - vertex_offset, vertex_offset]))
        pts.append(np.array([vertex_offset, 1 - vertex_offset]))
        pts.append(np.array([0.5, vertex_offset]))
        pts.append(np.array([0.5 * (1 - vertex_offset), 0.5 * (1 - vertex_offset)]))
        pts.append(np.array([vertex_offset, 0.5 * (1 - vertex_offset)]))
        pts.append(np.array([1/3, 1/3]))
        for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
            r = 0.25
            dx = r * np.cos(angle)
            dy = r * np.sin(angle)
            du = -dx / 2 + dy / H
            dv = dx / 2 + dy / H
            pt = np.array([1/3, 1/3]) + np.array([du, dv])
            if pt[0] > 0.01 and pt[1] > 0.01 and pt[0] + pt[1] < 0.98:
                pts.append(pt)
        
        if len(pts) == N:
            inits.append(np.array(pts))
    
    # Pattern 3: Grid-like arrangement
    for grid_size in [3, 4]:
        pts = []
        for i in range(grid_size):
            for j in range(grid_size - i):
                if len(pts) < N:
                    u = (i + 0.5) / grid_size * (1 - (j + 0.5) / grid_size)
                    v = (j + 0.5) / grid_size * (1 - (i + 0.5) / grid_size)
                    pts.append(np.array([u, v]))
        
        if len(pts) == N:
            inits.append(np.array(pts))
    
    return inits

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Smooth soft-min minimization with increasing beta
    for beta in [5.0, 20.0, 100.0, 300.0]:
        if time.time() - start_time > time_limit * 0.4:
            break
        try:
            result1 = minimize(
                soft_min_obj,
                bary_current.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if result1.success or result1.fun < -hard_min_obj(bary_current.flatten()):
                bary_current = clamp_bary(result1.x.reshape(N, 2))
        except Exception:
            pass
        
    # Stage 2: Hard min refinement with SLSQP
    if time.time() - start_time < time_limit * 0.7:
        try:
            result2 = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1500, 'ftol': 1e-16}
            )
            if result2.fun < hard_min_obj(bary_current.flatten()):
                bary_current = clamp_bary(result2.x.reshape(N, 2))
        except Exception:
            pass
            
    # Stage 3: Nelder-Mead for final polish
    if time.time() - start_time < time_limit:
        try:
            result3 = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
            )
            if result3.fun < hard_min_obj(bary_current.flatten()):
                bary_current = clamp_bary(result3.x.reshape(N, 2))
        except Exception:
            pass
        
    return bary_current

def coordinate_descent_refine(bary, max_iter=200, tol=1e-12):
    """
    Coordinate descent: optimize one point at a time to improve min area.
    """
    bary = bary.copy()
    best_val = -np.min(calc_areas_bary(bary))
    
    for iteration in range(max_iter):
        improved = False
        for i in range(N):
            old_pos = bary[i].copy()
            
            def obj_i(x, idx=i, base_bary=bary):
                bary_tmp = base_bary.copy()
                bary_tmp[idx] = x
                bary_tmp = clamp_bary(bary_tmp)
                areas = calc_areas_bary(bary_tmp)
                return -np.min(areas)
            
            result = minimize(
                obj_i,
                old_pos,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            
            if result.fun < -best_val - 1e-15:
                bary[i] = clamp_bary(result.x.reshape(1, 2))[0]
                best_val = -result.fun
                improved = True
        
        if not improved:
            break
    
    return bary, best_val

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

    # Generate diverse initial points
    rng = np.random.default_rng(123)
    inits = []
    
    # 1. Random starts
    for i in range(15):
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        inits.append(np.column_stack([u, v]))
        
    # 2. Structured starts from known patterns
    structured_inits = generate_structured_inits()
    inits.extend(structured_inits)
    
    # 3. Hexagonal grid
    for scale in [0.12, 0.15, 0.2]:
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
            for _ in range(3):
                idx = rng.choice(len(pts), N, replace=False)
                bary = np.array(pts)[idx]
                inits.append(bary)

    # Deduplicate inits
    unique_inits = []
    for init in inits:
        init_clamped = clamp_bary(init)
        is_unique = True
        for existing in unique_inits:
            if np.allclose(init_clamped, existing, atol=0.05):
                is_unique = False
                break
        if is_unique:
            unique_inits.append(init_clamped)
    
    inits = unique_inits
    
    # Refine each initial point with time budget
    time_per_init_base = 3.0
    for i, init in enumerate(inits):
        if time.time() > deadline - 15:
            break
            
        remaining_inits = len(inits) - i
        time_per_init = min(time_per_init_base, (deadline - time.time() - 15) / max(1, remaining_inits))
        if time_per_init < 0.5:
            time_per_init = 0.5
            
        refined = refine_local(init, time_limit=time_per_init)
        update_best(refined)
        
        # Also try coordinate descent on the refined result
        if time.time() < deadline - 20:
            cd_result, _ = coordinate_descent_refine(refined, max_iter=50)
            update_best(cd_result)

    # Global search with differential evolution
    if best_bary is not None and time.time() < deadline - 30:
        try:
            def de_objective(x):
                bary = x.reshape(N, 2)
                bary = clamp_bary(bary)
                return -np.min(calc_areas_bary(bary))
            
            bounds_de = [(0, 1)] * (N * 2)
            result_de = differential_evolution(
                de_objective,
                bounds_de,
                seed=42,
                maxiter=50,
                tol=1e-10,
                popsize=15,
                mutation=(0.3, 1.0),
                recombination=0.7,
                polish=False,
                update_interval=1,
                callback=None,
                init='sobol'
            )
            if result_de.success or result_de.fun < hard_min_obj(best_bary.flatten()):
                bary_de = clamp_bary(result_de.x.reshape(N, 2))
                if update_best(bary_de):
                    if time.time() < deadline - 10:
                        refined_de = refine_local(bary_de, time_limit=min(8.0, deadline - time.time() - 5))
                        update_best(refined_de)
        except Exception:
            pass
        
    # Final high-precision polish
    if best_bary is not None and time.time() < deadline - 8:
        try:
            bounds = [(0, 1)] * (N * 2)
            result_slsqp = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 3000, 'ftol': 1e-16}
            )
            if result_slsqp.fun < hard_min_obj(best_bary.flatten()):
                update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
        except Exception:
            pass
        
        if best_bary is not None and time.time() < deadline - 3:
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 8000, 'xatol': 1e-15, 'fatol': 1e-19, 'adaptive': True}
                )
                if result_nm.fun < hard_min_obj(best_bary.flatten()):
                    update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass

    cart = bary_to_cart(best_bary)
    
    return cart
