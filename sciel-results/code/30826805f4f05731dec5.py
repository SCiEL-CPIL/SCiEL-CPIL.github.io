import numpy as np
from scipy.optimize import differential_evolution, minimize
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

def disk_packing_init(seed=7):
    """
    Generate an initial configuration by maximizing minimum pairwise distance.
    This is a proxy for good area distribution.
    """
    rng = np.random.default_rng(seed)
    
    best_bary = None
    best_min_dist = -np.inf
    
    for attempt in range(10):
        # Start with random points
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        bary = np.column_stack([u, v])
        
        # Iteratively push points apart (simple repulsion) with annealing
        for it in range(500):
            bary = clamp_bary(bary)
            cart = bary_to_cart(bary)
            dists = np.linalg.norm(cart[:, None, :] - cart[None, :, :], axis=2)
            np.fill_diagonal(dists, np.inf)
            
            forces = np.zeros_like(cart)
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    d = dists[i, j]
                    if d < 1e-6:
                        continue
                    forces[i] += (cart[i] - cart[j]) / (d * d)
            
            step_size = 0.02 * (1.0 - it / 500.0) + 0.002
            bary += step_size * forces
            bary = clamp_bary(bary)
        
        cart = bary_to_cart(bary)
        dists = np.linalg.norm(cart[:, None, :] - cart[None, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best_bary = bary.copy()
    
    return best_bary

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration.
    """
    best_bary = None
    best_val = -np.inf
    
    for scale in [0.2, 0.25, 0.3, 0.35, 0.4]:
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

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages with graduated soft-min.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Graduated soft-min minimization
    # We increase beta to make the surrogate sharper, effectively solving the hard min
    beta_schedule = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    
    for beta in beta_schedule:
        if time.time() - start_time > time_limit * 0.5:
            break
        try:
            result = minimize(
                soft_min_obj,
                bary_current.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            new_bary = clamp_bary(result.x.reshape(N, 2))
            old_val = -hard_min_obj(bary_current.flatten())
            new_val = -hard_min_obj(new_bary.flatten())
            if new_val > old_val:
                bary_current = new_bary
        except Exception:
            pass
    
    if time.time() - start_time > time_limit * 0.7:
        return bary_current

    # Stage 2: Nelder-Mead on hard min
    # NM is robust to non-smoothness
    try:
        result_nm = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-15, 'adaptive': True}
        )
        new_bary = clamp_bary(result_nm.x.reshape(N, 2))
        old_val = -hard_min_obj(bary_current.flatten())
        new_val = -hard_min_obj(new_bary.flatten())
        if new_val > old_val:
            bary_current = new_bary
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.9:
        return bary_current

    # Stage 3: SLSQP polish
    try:
        result_slsqp = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-15}
        )
        new_bary = clamp_bary(result_slsqp.x.reshape(N, 2))
        old_val = -hard_min_obj(bary_current.flatten())
        new_val = -hard_min_obj(new_bary.flatten())
        if new_val > old_val:
            bary_current = new_bary
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

    # 1. Initial Heuristics
    # Disk packing is usually a strong start
    disk_init = disk_packing_init(seed=42)
    refined_disk = refine_local(disk_init, time_limit=10.0)
    update_best(refined_disk)
    
    # Hexagonal is another good structure
    hex_init = generate_hexagonal_init()
    refined_hex = refine_local(hex_init, time_limit=10.0)
    update_best(refined_hex)
    
    # 2. Global Search with Differential Evolution
    # DE is effective for non-convex problems. We allocate a significant chunk of time here.
    if time.time() < deadline - 60:
        try:
            def de_obj(x):
                bary = x.reshape(N, 2)
                bary = clamp_bary(bary)
                areas = calc_areas_bary(bary)
                return -np.min(areas)
            
            bounds = [(0, 1)] * (N * 2)
            # Use a relatively large population and iterations
            # popsize=20, maxiter=50 is a good balance for 150s limit if local refine is fast
            result_de = differential_evolution(
                de_obj,
                bounds,
                seed=42,
                maxiter=50,
                popsize=20,
                tol=1e-8,
                polish=False,
                init='sobol' # Sobol sequence for better coverage
            )
            
            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined_de = refine_local(de_bary, time_limit=10.0)
            update_best(refined_de)
        except Exception:
            pass

    # 3. Multi-start Local Refinement
    # Use remaining time for random restarts and perturbations of the best solution
    restart_count = 0
    while time.time() < deadline - 15:
        restart_count += 1
        
        if restart_count % 3 == 0 and best_bary is not None:
            # Perturb best solution
            sigma = 0.05
            rng = np.random.default_rng(1000 + restart_count)
            noise = rng.normal(0, sigma, (N, 2))
            init = clamp_bary(best_bary + noise)
        elif restart_count % 3 == 1:
            # Random start
            rng = np.random.default_rng(2000 + restart_count)
            u = rng.random(N)
            v = rng.random(N) * (1 - u)
            init = np.column_stack([u, v])
        else:
            # Another disk packing attempt with different seed
            init = disk_packing_init(seed=100 + restart_count)
        
        # Shorter time limit for these restarts
        tl = 3.0
        refined = refine_local(init, time_limit=tl)
        update_best(refined)
        
        # If we have a very good solution, try a quick polish on it occasionally
        if best_bary is not None and restart_count % 5 == 0 and time.time() < deadline - 10:
            # Quick NM polish
            try:
                result_quick = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                update_best(clamp_bary(result_quick.x.reshape(N, 2)))
            except Exception:
                pass

    # 4. Final High-Precision Polish
    if best_bary is not None and time.time() < deadline - 2:
        try:
            # Final NM with tight tolerances
            result_final = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            update_best(clamp_bary(result_final.x.reshape(N, 2)))
        except Exception:
            pass
        
        if best_bary is not None and time.time() < deadline - 0.5:
            try:
                # Final SLSQP
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

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
