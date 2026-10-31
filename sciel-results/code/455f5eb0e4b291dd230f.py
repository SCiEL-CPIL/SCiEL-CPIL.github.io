import numpy as np
from scipy.optimize import minimize, differential_evolution
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

def clamp_bary(bary, eps=1e-10):
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
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

def hard_min_obj(bary_flat):
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def soft_min_obj(bary_flat, beta=50.0):
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

def generate_hexagonal_init(jitter=0.0):
    best_bary = None
    best_val = -np.inf
    
    rng = np.random.default_rng(42)
    
    for scale in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
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
                if len(pts) == N:
                    bary = np.array(pts, dtype=float)
                else:
                    idx = rng.choice(len(pts), N, replace=False)
                    bary = np.array(pts, dtype=float)[idx]
                
                if jitter > 0:
                    bary = bary + rng.normal(0, jitter, bary.shape)
                    bary = clamp_bary(bary)
                
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
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def generate_far_point_init(seed=None):
    rng = np.random.default_rng(seed)
    
    n_grid = 20
    u_grid, v_grid = np.meshgrid(np.linspace(0, 1, n_grid), np.linspace(0, 1, n_grid))
    u_grid = u_grid.flatten()
    v_grid = v_grid.flatten()
    
    valid = (u_grid >= 0) & (v_grid >= 0) & (u_grid + v_grid <= 1)
    u_valid = u_grid[valid]
    v_valid = v_grid[valid]
    
    if len(u_valid) < N:
        return generate_random_init(seed)
    
    idxs = [rng.choice(len(u_valid))]
    
    for _ in range(N - 1):
        last_u, last_v = u_valid[idxs[-1]], v_valid[idxs[-1]]
        dists = (u_valid - last_u)**2 + (v_valid - last_v)**2
        for i in idxs:
            dists[i] = -1
        next_idx = np.argmax(dists)
        idxs.append(next_idx)
        
    bary = np.column_stack([u_valid[idxs], v_valid[idxs]])
    return bary

def refine_local(bary_init, time_limit=5.0):
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Coarse soft-min
    try:
        result1 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(20.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result1.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result1.x.reshape(N, 2))
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.2:
        return bary_current

    # Stage 2: Medium soft-min
    try:
        result2 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(100.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-9}
        )
        if result2.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result2.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.4:
        return bary_current

    # Stage 3: Sharp soft-min
    try:
        result3 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(500.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if result3.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result3.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.6:
        return bary_current

    # Stage 4: Nelder-Mead on hard min
    try:
        result4 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
        )
        bary_current = clamp_bary(result4.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.8:
        return bary_current

    # Stage 5: SLSQP polish
    try:
        result5 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-14}
        )
        bary_current = clamp_bary(result5.x.reshape(N, 2))
    except Exception:
        pass
        
    return bary_current

def push_apart(bary, time_limit=2.0):
    """
    Identify the critical triangle (smallest area) and push its vertices apart.
    """
    start_time = time.time()
    bary = bary.copy()
    
    for iteration in range(50):
        if time.time() - start_time > time_limit:
            break
        
        areas = calc_areas_bary(bary)
        min_idx = np.argmin(areas)
        min_area = areas[min_idx]
        
        if min_area > 1e-10:
            break
        
        # Get the three points of the critical triangle
        i, j, k = tri_idx[min_idx]
        
        # Compute centroid of the critical triangle
        centroid = (bary[i] + bary[j] + bary[k]) / 3.0
        
        # Push each vertex away from centroid
        push_factor = 0.01
        for idx in [i, j, k]:
            direction = bary[idx] - centroid
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction = direction / norm
                bary[idx] += push_factor * direction
        
        bary = clamp_bary(bary)
    
    return bary

def coordinate_descent_refine(bary_init, time_limit=4.0):
    start_time = time.time()
    bary = bary_init.copy()
    
    for _ in range(5):
        if time.time() - start_time > time_limit:
            break
        for i in range(N):
            if time.time() - start_time > time_limit:
                break
            
            def obj_1d(theta):
                u_new = bary[:, 0].copy()
                v_new = bary[:, 1].copy()
                u_new[i] = bary[i, 0] + theta[0]
                v_new[i] = bary[i, 1] + theta[1]
                
                bary_new = np.column_stack([u_new, v_new])
                bary_new = clamp_bary(bary_new)
                areas = calc_areas_bary(bary_new)
                return -np.min(areas)
            
            try:
                result = minimize(
                    obj_1d,
                    [0.0, 0.0],
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-14}
                )
                bary[i, 0] += result.x[0]
                bary[i, 1] += result.x[1]
                bary = clamp_bary(bary)
            except Exception:
                pass
    
    return bary

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

    # 1. Try multiple hexagonal inits with jitter
    for jitter in [0.0, 0.01, 0.02, 0.03]:
        if time.time() > deadline - 30:
            break
        hex_init = generate_hexagonal_init(jitter=jitter)
        refined = refine_local(hex_init, time_limit=5.0)
        update_best(refined)
        
        if time.time() > deadline - 25:
            break
        # Push-apart heuristic
        pushed = push_apart(refined, time_limit=2.0)
        update_best(pushed)
        
        if time.time() > deadline - 20:
            break
        # Coordinate descent
        cd = coordinate_descent_refine(refined, time_limit=3.0)
        update_best(cd)

    # 2. Try far-point init
    if time.time() < deadline - 25:
        fp_init = generate_far_point_init(seed=42)
        refined_fp = refine_local(fp_init, time_limit=5.0)
        update_best(refined_fp)
        
        if time.time() < deadline - 20:
            pushed_fp = push_apart(refined_fp, time_limit=2.0)
            update_best(pushed_fp)
    
    # 3. Global search with Differential Evolution
    if time.time() < deadline - 30:
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
                polish=False
            )

            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined_de = refine_local(de_bary, time_limit=6.0)
            update_best(refined_de)
            
            if time.time() < deadline - 20:
                pushed_de = push_apart(refined_de, time_limit=2.0)
                update_best(pushed_de)
        except Exception:
            pass

    # 4. Multi-start with diverse strategies
    rng = np.random.default_rng(123)
    restart_count = 0
    while time.time() < deadline - 15:
        restart_count += 1
        if best_bary is not None and restart_count % 3 == 0:
            # Perturbation of best
            sigma = 0.03
            noise = rng.normal(0, sigma, best_bary.shape)
            init = clamp_bary(best_bary + noise)
        elif restart_count % 4 == 0:
            # Far-point with different seed
            init = generate_far_point_init(seed=5000 + restart_count)
        else:
            # Fresh random
            init = generate_random_init(seed=5000 + restart_count)
        
        if restart_count % 5 == 0:
            refined = coordinate_descent_refine(init, time_limit=3.0)
        else:
            refined = refine_local(init, time_limit=3.0)
        update_best(refined)
        
        if time.time() < deadline - 10 and restart_count % 2 == 0:
            pushed = push_apart(refined, time_limit=1.5)
            update_best(pushed)

    # 5. Final high-precision polish
    if best_bary is not None and time.time() < deadline - 5:
        try:
            bounds = [(0, 1)] * (N * 2)
            result_slsqp = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-15}
            )
            update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
        except Exception:
            pass
        
        if best_bary is not None and time.time() < deadline - 3:
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
        
        if best_bary is not None and time.time() < deadline - 2:
            # Final push-apart
            pushed = push_apart(best_bary, time_limit=1.5)
            update_best(pushed)
        
        if best_bary is not None and time.time() < deadline - 1:
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

    cart = bary_to_cart(best_bary)
    return cart
