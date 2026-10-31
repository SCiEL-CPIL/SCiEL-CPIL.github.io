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

def epigraph_obj(x, beta=100.0):
    """
    Epigraph formulation: x = [bary_flat (22), t (1)]
    Minimize -t subject to soft constraints: all areas >= t (softly).
    """
    bary = x[:2*N].reshape(N, 2)
    t = x[2*N]
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Penalty for areas < t
    diff = areas - t
    if np.any(diff < -1e-10):
        # Large penalty if any area is significantly below t
        penalty = np.sum(np.maximum(0, -diff) ** 2) * 1e6
        return -t + penalty
    
    # Soft constraint via log-sum-exp
    if np.any(areas < 1e-15):
        return 1e10
    
    scaled = -beta * diff
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        return -t + 1e6 * np.sum(np.maximum(0, -diff) ** 2)
        
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    # log_sum / beta should be <= 0 for feasibility; penalty if positive
    penalty = max(0, log_sum / beta) * 1e4
    return -t + penalty

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

def generate_far_point_init(seed=None):
    """
    Generate an initial configuration using farthest-point sampling on a grid.
    This creates well-spread initial points.
    """
    rng = np.random.default_rng(seed)
    
    # Create a fine grid in barycentric coordinates
    n_grid = 15
    u_grid, v_grid = np.meshgrid(np.linspace(0, 1, n_grid), np.linspace(0, 1, n_grid))
    u_grid = u_grid.flatten()
    v_grid = v_grid.flatten()
    
    # Filter valid points
    valid = (u_grid >= 0) & (v_grid >= 0) & (u_grid + v_grid <= 1)
    u_valid = u_grid[valid]
    v_valid = v_grid[valid]
    
    if len(u_valid) < N:
        return generate_random_init(seed)
    
    # Farthest-point sampling
    # Start with a random point
    idxs = [rng.choice(len(u_valid))]
    
    for _ in range(N - 1):
        last_u, last_v = u_valid[idxs[-1]], v_valid[idxs[-1]]
        dists = (u_valid - last_u)**2 + (v_valid - last_v)**2
        # Exclude already chosen
        for i in idxs:
            dists[i] = -1
        # Pick the farthest
        next_idx = np.argmax(dists)
        idxs.append(next_idx)
        
    bary = np.column_stack([u_valid[idxs], v_valid[idxs]])
    return bary

def coordinate_descent_refine(bary_init, time_limit=5.0):
    """
    Coordinate descent refinement: optimize each point's coordinates one at a time.
    This can escape some local minima that gradient-based methods get stuck in.
    """
    start_time = time.time()
    bary = bary_init.copy()
    
    # Pre-compute all other points for each point
    def optimize_point(i, bary_current):
        """Optimize point i while keeping others fixed."""
        u = bary_current[:, 0].copy()
        v = bary_current[:, 1].copy()
        
        def obj_1d(theta):
            # theta = [du, dv] where new point = old + theta
            u_new = u.copy()
            v_new = v.copy()
            u_new[i] = u[i] + theta[0]
            v_new[i] = v[i] + theta[1]
            
            bary_new = np.column_stack([u_new, v_new])
            bary_new = clamp_bary(bary_new)
            areas = calc_areas_bary(bary_new)
            return -np.min(areas)
        
        # Use Nelder-Mead for the 2D subproblem
        try:
            result = minimize(
                obj_1d,
                [0.0, 0.0],
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            return result.x
        except Exception:
            return [0.0, 0.0]
    
    # Multiple passes
    for _ in range(3):
        if time.time() - start_time > time_limit:
            break
        for i in range(N):
            if time.time() - start_time > time_limit:
                break
            delta = optimize_point(i, bary)
            bary[i, 0] += delta[0]
            bary[i, 1] += delta[1]
            bary = clamp_bary(bary)
    
    return bary

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages with graduated soft-min.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Coarse soft-min (low beta, more iterations)
    try:
        result1 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(20.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result1.success or result1.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result1.x.reshape(N, 2))
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.25:
        return bary_current

    # Stage 2: Medium soft-min (increasing beta)
    try:
        result2 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(50.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-9}
        )
        if result2.success or result2.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result2.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.45:
        return bary_current

    # Stage 3: Sharp soft-min (high beta, close to hard min)
    try:
        result3 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(200.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if result3.success or result3.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result3.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.65:
        return bary_current

    # Stage 4: Nelder-Mead on hard min
    try:
        result4 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1500, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
        )
        bary_current = clamp_bary(result4.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.85:
        return bary_current

    # Stage 5: SLSQP polish
    try:
        result5 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14}
        )
        bary_current = clamp_bary(result5.x.reshape(N, 2))
    except Exception:
        pass
        
    return bary_current

def refine_epigraph(bary_init, t_init=None, time_limit=5.0):
    """
    Refine using epigraph formulation with auxiliary variable t.
    """
    start_time = time.time()
    
    if t_init is None:
        areas = calc_areas_bary(bary_init)
        t_init = np.min(areas)
    
    x0 = np.concatenate([bary_init.flatten(), [t_init]])
    
    # Bounds for x: bary in [0,1], t in [0, 0.1] (max possible area is ~0.086 for unit triangle)
    bounds = [(0, 1)] * (N * 2) + [(0, 0.1)]
    
    current_x = x0.copy()
    
    # Stage 1: Coarse epigraph
    try:
        result1 = minimize(
            epigraph_obj,
            current_x,
            args=(50.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result1.success:
            current_x = result1.x
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.3:
        return current_x[:2*N].reshape(N, 2)

    # Stage 2: Medium epigraph
    try:
        result2 = minimize(
            epigraph_obj,
            current_x,
            args=(100.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-9}
        )
        if result2.success:
            current_x = result2.x
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.6:
        return current_x[:2*N].reshape(N, 2)

    # Stage 3: Sharp epigraph
    try:
        result3 = minimize(
            epigraph_obj,
            current_x,
            args=(300.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if result3.success:
            current_x = result3.x
    except Exception:
        pass
        
    return current_x[:2*N].reshape(N, 2)

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
    refined_hex = refine_local(hex_init, time_limit=8.0)
    update_best(refined_hex)
    
    # Also try coordinate descent on hex init
    if time.time() < deadline - 30:
        cd_hex = coordinate_descent_refine(hex_init, time_limit=4.0)
        update_best(cd_hex)

    # Also try epigraph refinement on hex init
    if time.time() < deadline - 25:
        ep_refined = refine_epigraph(hex_init, time_limit=5.0)
        update_best(ep_refined)

    # 2. Try Far-Point Init with local refinement
    fp_init = generate_far_point_init(seed=42)
    refined_fp = refine_local(fp_init, time_limit=6.0)
    update_best(refined_fp)
    
    # Coordinate descent on far-point init
    if time.time() < deadline - 20:
        cd_fp = coordinate_descent_refine(fp_init, time_limit=4.0)
        update_best(cd_fp)

    # 3. Global Search with Differential Evolution (increased population and iterations)
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
                popsize=20,
                tol=1e-6,
                polish=False
            )

            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined_de = refine_local(de_bary, time_limit=6.0)
            update_best(refined_de)
            
            # Also try coordinate descent on DE result
            if time.time() < deadline - 25:
                cd_de = coordinate_descent_refine(de_bary, time_limit=4.0)
                update_best(cd_de)
        except Exception:
            pass

    # 4. Multi-start: random + perturbed restarts with local refinement
    rng = np.random.default_rng(123)
    restart_count = 0
    while time.time() < deadline - 12:
        restart_count += 1
        if best_bary is not None and restart_count % 2 == 0:
            # Perturbation of best solution
            init = generate_perturbed_init(best_bary, sigma=0.04, seed=2000 + restart_count)
        elif restart_count % 3 == 0:
            # Far-point init with different seed
            init = generate_far_point_init(seed=3000 + restart_count)
        else:
            # Fresh random start
            init = generate_random_init(seed=3000 + restart_count)
        
        # Alternate between local refinement, coordinate descent, and epigraph refinement
        if restart_count % 4 == 0:
            refined = coordinate_descent_refine(init, time_limit=3.0)
        elif restart_count % 3 == 0:
            refined = refine_epigraph(init, time_limit=3.0)
        else:
            refined = refine_local(init, time_limit=3.0)
        update_best(refined)

    # 5. Final high-precision polish
    if best_bary is not None and time.time() < deadline - 5:
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
        
        if best_bary is not None and time.time() < deadline - 3:
            # Try coordinate descent polish
            try:
                cd_polish = coordinate_descent_refine(best_bary, time_limit=2.0)
                update_best(cd_polish)
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
