import numpy as np
from scipy.optimize import differential_evolution, minimize, minimize_scalar
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
    """Clamp barycentric coordinates to the strictly valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    u = np.clip(u, eps, 1.0)
    v = np.clip(v, eps, 1.0)
    
    sums = u + v
    invalid = sums > (1.0 - eps)
    if np.any(invalid):
        scale = (1.0 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    u = np.clip(u, eps, 1.0)
    v = np.clip(v, eps, 1.0)
    
    return np.column_stack([u, v])

def hard_min_obj(bary_flat):
    """Objective using hard min."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def soft_min_obj(bary_flat, beta=100.0):
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

def epigraph_obj(x):
    """
    Epigraph formulation: minimize -t subject to areas >= t.
    x = [bary_0, ..., bary_{2N-1}, t]
    """
    bary = x[:2*N].reshape(N, 2)
    t = x[2*N]
    
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Penalty for violating areas >= t
    violations = np.maximum(0, t - areas)
    penalty = 1e6 * np.sum(violations**2)
    
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

def generate_boundary_interior_init(seed=None):
    """
    Generate initialization with some points on boundary and some interior.
    """
    rng = np.random.default_rng(seed)
    bary = np.zeros((N, 2))
    
    # Place 3 points near vertices
    bary[0] = [0.05, 0.05]
    bary[1] = [0.95, 0.05]
    bary[2] = [0.05, 0.95]
    
    # Place 4 points on edges
    for i in range(3, 7):
        edge = (i - 3) % 3
        t = rng.uniform(0.2, 0.8)
        if edge == 0:  # Bottom edge
            bary[i] = [t, 0.01]
        elif edge == 1:  # Right edge
            bary[i] = [0.01, t]
        else:  # Left edge
            bary[i] = [1.0 - t - 0.01, 0.01]
    
    # Place 4 points interior
    for i in range(7, 11):
        u = rng.uniform(0.1, 0.8)
        v = rng.uniform(0.1, 0.8 * (1 - u))
        bary[i] = [u, v]
    
    return clamp_bary(bary)

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Smooth soft-min minimization with L-BFGS-B
    try:
        result1 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(100.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if result1.success or result1.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result1.x.reshape(N, 2))
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.3:
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
        
    if time.time() - start_time > time_limit * 0.6:
        return bary_current
    
    # Stage 3: SLSQP polish
    try:
        result3 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14}
        )
        bary_current = clamp_bary(result3.x.reshape(N, 2))
    except Exception:
        pass
        
    # Stage 4: Epigraph formulation with SLSQP
    if time.time() - start_time < time_limit * 0.9:
        try:
            # Initialize t with current min area
            areas = calc_areas_bary(bary_current)
            t_init = np.min(areas) * 0.99
            x_init = np.concatenate([bary_current.flatten(), [t_init]])
            
            bounds_ep = [(0, 1)] * (N * 2) + [(0, 0.1)]
            
            result4 = minimize(
                epigraph_obj,
                x_init,
                method='SLSQP',
                bounds=bounds_ep,
                options={'maxiter': 300, 'ftol': 1e-14}
            )
            bary_ep = clamp_bary(result4.x[:2*N].reshape(N, 2))
            val_ep = -hard_min_obj(bary_ep.flatten())
            val_cur = -hard_min_obj(bary_current.flatten())
            if val_ep > val_cur:
                bary_current = bary_ep
        except Exception:
            pass
        
    return bary_current

def coordinate_descent_polish(bary, time_limit=2.0):
    """
    Coordinate descent polish: optimize each point's (u, v) while fixing others.
    """
    start_time = time.time()
    bary = bary.copy()
    
    for _ in range(100):  # Max 100 sweeps
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        for i in range(N):
            # Optimize point i
            def obj_i(params):
                bary_temp = bary.copy()
                bary_temp[i, 0] = params[0]
                bary_temp[i, 1] = params[1]
                bary_temp = clamp_bary(bary_temp)
                areas = calc_areas_bary(bary_temp)
                return -np.min(areas)
            
            # Use Nelder-Mead for 2D optimization
            try:
                result = minimize(
                    obj_i,
                    bary[i],
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-14}
                )
                new_point = clamp_bary(result.x.reshape(1, 2))[0]
                if result.fun < obj_i(bary[i]):
                    bary[i] = new_point
                    improved = True
            except Exception:
                pass
        
        if not improved:
            break
    
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

    # 1. Try Hexagonal Init
    hex_init = generate_hexagonal_init()
    refined_hex = refine_local(hex_init, time_limit=8.0)
    update_best(refined_hex)
    
    # 2. Try Boundary-Interior Init
    bi_init = generate_boundary_interior_init(seed=123)
    refined_bi = refine_local(bi_init, time_limit=6.0)
    update_best(refined_bi)
    
    # 3. Global Search with Differential Evolution
    if time.time() < deadline - 35:
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
                maxiter=40,
                popsize=15,
                tol=1e-6,
                polish=False
            )
            
            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined_de = refine_local(de_bary, time_limit=8.0)
            update_best(refined_de)
        except Exception:
            pass
    
    # 4. Multi-start: random + perturbed restarts with local refinement
    rng = np.random.default_rng(123)
    restart_count = 0
    while time.time() < deadline - 15:
        restart_count += 1
        if best_bary is not None and restart_count % 3 == 0:
            # Perturbation of best solution
            init = generate_perturbed_init(best_bary, sigma=0.03, seed=2000 + restart_count)
        elif best_bary is not None and restart_count % 3 == 1:
            # Boundary-interior perturbation
            init = generate_boundary_interior_init(seed=4000 + restart_count)
        else:
            # Fresh random start
            init = generate_random_init(seed=3000 + restart_count)
        
        refined = refine_local(init, time_limit=4.0)
        update_best(refined)

    # 5. Final high-precision polish
    if best_bary is not None and time.time() < deadline - 10:
        # Coordinate descent polish
        polished_cd = coordinate_descent_polish(best_bary, time_limit=5.0)
        update_best(polished_cd)
    
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
