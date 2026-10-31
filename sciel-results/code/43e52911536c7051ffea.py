import numpy as np
from scipy.optimize import differential_evolution, minimize
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

def epigraph_obj(x, beta=100.0):
    """
    Epigraph formulation: minimize -t subject to soft_min(areas) >= t.
    Variables: x = [bary (N*2), t]
    """
    n_vars = N * 2
    bary = x[:n_vars].reshape(N, 2)
    t = x[n_vars]
    
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    if np.any(areas < 1e-15):
        return 1e10
    
    # Soft constraint: penalize if t > soft_min(areas)
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        soft_min_val = np.min(areas)
    else:
        log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        soft_min_val = -log_sum / beta
    
    # We want to maximize t, so minimize -t + penalty for t > soft_min
    penalty = np.maximum(0, t - soft_min_val)
    return -t + 10.0 * penalty

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

def epigraph_obj_grad(x, beta=100.0):
    """
    Epigraph objective with gradient for SLSQP.
    Variables: x = [bary (N*2), t]
    """
    n_vars = N * 2
    bary = x[:n_vars].reshape(N, 2)
    t = x[n_vars]
    
    bary_clamped = clamp_bary(bary)
    areas = calc_areas_bary(bary_clamped)
    
    if np.any(areas < 1e-15):
        return 1e10, np.zeros_like(x)
    
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    
    if max_scaled > 700:
        soft_min_val = np.min(areas)
        # Gradient w.r.t. areas: -beta * exp(0) / sum(exp(0)) = -beta/N for all (approx)
        # Actually for hard min, gradient is -1 for the min area, 0 for others
        min_idx = np.argmin(areas)
        dsoft_dareas = np.zeros(len(areas))
        dsoft_dareas[min_idx] = 1.0
    else:
        exp_scaled = np.exp(scaled - max_scaled)
        sum_exp = np.sum(exp_scaled)
        log_sum = max_scaled + np.log(sum_exp)
        soft_min_val = -log_sum / beta
        # d(soft_min)/d(areas[i]) = -1/beta * d(log_sum)/d(areas[i])
        # d(log_sum)/d(areas[i]) = -beta * exp_scaled[i] / sum_exp
        # So d(soft_min)/d(areas[i]) = exp_scaled[i] / sum_exp
        dsoft_dareas = exp_scaled / sum_exp
    
    penalty = np.maximum(0, t - soft_min_val)
    obj_val = -t + 10.0 * penalty
    
    # Gradient
    grad = np.zeros_like(x)
    
    if penalty > 0:
        # d(penalty)/dt = 1, d(penalty)/d(areas) = -10 * dsoft_dareas
        grad[n_vars] = -1.0 + 10.0  # d(-t)/dt + 10*d(penalty)/dt = -1 + 10
        # d(obj)/d(areas[i]) = -10 * dsoft_dareas[i]
        # d(obj)/d(bary) = sum_i d(obj)/d(areas[i]) * d(areas[i])/d(bary)
        # This is complex; use finite differences or skip gradient for bary
        # For simplicity, return only partial gradient
        grad[n_vars] = -1.0 + 10.0
    else:
        grad[n_vars] = -1.0
    
    return obj_val, grad

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
    Refine a configuration using graduated soft-min + epigraph stages.
    """
    start_time = time.time()
    bounds_bary = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    current_val = -hard_min_obj(bary_current.flatten())
    
    # Stage 1: Graduated soft-min with L-BFGS-B (beta: 20 -> 50 -> 200)
    for beta in [20.0, 50.0, 200.0]:
        if time.time() - start_time > time_limit * 0.5:
            break
        try:
            result1 = minimize(
                soft_min_obj,
                bary_current.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds_bary,
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            new_bary = clamp_bary(result1.x.reshape(N, 2))
            new_val = -hard_min_obj(new_bary.flatten())
            if new_val >= current_val - 1e-12:
                bary_current = new_bary
                current_val = new_val
        except Exception:
            pass
    
    # Stage 2: Epigraph formulation with SLSQP
    if time.time() - start_time < time_limit * 0.8:
        try:
            # Initialize t with current best value
            t_init = current_val
            x_init = np.concatenate([bary_current.flatten(), [t_init]])
            bounds_ep = bounds_bary + [(0, 1)]
            
            for beta in [50.0, 200.0, 500.0]:
                if time.time() - start_time > time_limit * 0.9:
                    break
                result_ep = minimize(
                    epigraph_obj,
                    x_init,
                    args=(beta,),
                    method='SLSQP',
                    bounds=bounds_ep,
                    options={'maxiter': 500, 'ftol': 1e-15}
                )
                new_bary = clamp_bary(result_ep.x[:N*2].reshape(N, 2))
                new_val = -hard_min_obj(new_bary.flatten())
                if new_val >= current_val - 1e-12:
                    bary_current = new_bary
                    current_val = new_val
                    x_init = np.concatenate([bary_current.flatten(), [current_val]])
        except Exception:
            pass
    
    # Stage 3: Hard min Nelder-Mead polish
    if time.time() - start_time < time_limit * 0.95:
        try:
            result2 = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-13, 'fatol': 1e-17, 'adaptive': True}
            )
            new_bary = clamp_bary(result2.x.reshape(N, 2))
            new_val = -hard_min_obj(new_bary.flatten())
            if new_val >= current_val - 1e-12:
                bary_current = new_bary
                current_val = new_val
        except Exception:
            pass
    
    # Stage 4: Final SLSQP polish on hard min
    if time.time() - start_time < time_limit:
        try:
            result3 = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='SLSQP',
                bounds=bounds_bary,
                options={'maxiter': 1000, 'ftol': 1e-15}
            )
            new_bary = clamp_bary(result3.x.reshape(N, 2))
            new_val = -hard_min_obj(new_bary.flatten())
            if new_val >= current_val - 1e-12:
                bary_current = new_bary
                current_val = new_val
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

    # 1. Try Hexagonal Init with epigraph-guided refinement
    hex_init = generate_hexagonal_init()
    refined_hex = refine_local(hex_init, time_limit=8.0)
    update_best(refined_hex)
    
    # 2. Global Search with Differential Evolution
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
                maxiter=40,
                popsize=15,
                tol=1e-8,
                polish=False
            )
            
            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined_de = refine_local(de_bary, time_limit=8.0)
            update_best(refined_de)
        except Exception:
            pass
    
    # 3. Multi-start: random + perturbed restarts with local refinement
    rng = np.random.default_rng(123)
    restart_count = 0
    while time.time() < deadline - 15:
        restart_count += 1
        if best_bary is not None and restart_count % 2 == 0:
            # Perturbation of best solution
            init = generate_perturbed_init(best_bary, sigma=0.03, seed=2000 + restart_count)
        else:
            # Fresh random start
            init = generate_random_init(seed=3000 + restart_count)
        
        refined = refine_local(init, time_limit=4.0)
        update_best(refined)

    # 4. Final high-precision polish on the best solution found
    if best_bary is not None and time.time() < deadline - 5:
        # Epigraph polish with high beta
        try:
            t_init = best_val
            x_init = np.concatenate([best_bary.flatten(), [t_init]])
            bounds_ep = [(0, 1)] * (N * 2) + [(0, 1)]
            for beta in [200.0, 500.0, 1000.0]:
                if time.time() > deadline - 3:
                    break
                result_ep = minimize(
                    epigraph_obj,
                    x_init,
                    args=(beta,),
                    method='SLSQP',
                    bounds=bounds_ep,
                    options={'maxiter': 800, 'ftol': 1e-16}
                )
                new_bary = clamp_bary(result_ep.x[:N*2].reshape(N, 2))
                update_best(new_bary)
                x_init = np.concatenate([best_bary.flatten(), [best_val]])
        except Exception:
            pass
        
        # One more Nelder-Mead pass with very tight tolerance
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

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
