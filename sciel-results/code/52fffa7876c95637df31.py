import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 115.0  # Safety margin for 120s limit

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

def cart_to_bary(cart):
    """Convert Cartesian (x, y) to barycentric coords (u, v)."""
    x = cart[:, 0]
    y = cart[:, 1]
    v = (x - 0.5 * y / H)  # This is actually w
    # Let's re-derive:
    # x = v + 0.5*w, y = w*H => w = y/H
    # v = x - 0.5*w = x - 0.5*y/H
    # u = 1 - v - w = 1 - (x - 0.5*y/H) - y/H = 1 - x + 0.5*y/H - y/H = 1 - x - 0.5*y/H
    w = y / H
    v = x - 0.5 * w
    u = 1.0 - v - w
    return np.column_stack([u, v])

def clamp_bary(bary, eps=1e-10):
    """Clamp barycentric coordinates to the valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    # First clamp u and v individually
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)
    
    # Then enforce u + v <= 1
    sums = u + v
    invalid = sums > 1.0
    if np.any(invalid):
        scale = 1.0 / sums[invalid]
        u[invalid] *= scale
        v[invalid] *= scale
    
    # Final safety clamp
    u = np.clip(u, eps, 1.0 - eps)
    v = np.clip(v, eps, 1.0 - eps)
    
    # Re-check constraint after clamping
    sums = u + v
    invalid = sums > (1.0 - eps)
    if np.any(invalid):
        scale = (1.0 - eps) / sums[invalid]
        u[invalid] *= scale
        v[invalid] *= scale
    
    return np.column_stack([u, v])

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

def get_min_area(bary):
    """Get the minimum area of any triangle formed by the points."""
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return np.min(areas)

def hard_min_obj(bary_flat):
    """Objective: negative of min area (to minimize)."""
    bary = bary_flat.reshape(N, 2)
    return -get_min_area(bary)

def soft_min_obj(bary_flat, beta=100.0):
    """
    Smooth surrogate for the hard min via log-sum-exp.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Avoid numerical issues
    min_area = np.min(areas)
    if min_area < 1e-15:
        return 1e10
    
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    
    # Handle overflow
    if max_scaled > 700:
        return -min_area
    
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    return -soft_min

def epigraph_obj(params):
    """
    Epigraph formulation: minimize -t subject to areas >= t.
    params = [bary(2N), t]
    """
    bary = params[:2*N].reshape(N, 2)
    t = params[2*N]
    
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Penalty for violating areas >= t
    violations = np.maximum(0, t - areas)
    penalty = np.sum(violations ** 2)
    
    # We want to maximize t, so minimize -t + penalty
    return -t + 10.0 * penalty

def epigraph_smooth_obj(params, beta=100.0):
    """
    Smooth epigraph formulation using soft constraints.
    """
    bary = params[:2*N].reshape(N, 2)
    t = params[2*N]
    
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Soft penalty: -log(areas - t + eps) style, but simpler quadratic
    # Use: sum(max(0, t - areas)^2) but smoothed
    diff = t - areas
    # Smooth max via softplus
    beta_soft = 50.0
    soft_max = (1.0/beta_soft) * np.log(1.0 + np.exp(beta_soft * diff))
    penalty = np.sum(soft_max ** 2)
    
    return -t + 5.0 * penalty

def generate_hexagonal_init():
    """Generate a hexagonal lattice-like initial configuration."""
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
            rng = np.random.default_rng(42)
            for _ in range(20):
                if len(pts) == N:
                    bary = np.array(pts)
                else:
                    idx = rng.choice(len(pts), N, replace=False)
                    bary = np.array(pts)[idx]
                val = get_min_area(bary)
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
    Refine a configuration using multi-stage optimization.
    Stage 1-3: Graduated soft-min via L-BFGS-B
    Stage 4: Epigraph formulation
    Stage 5: Final Nelder-Mead polish
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Smooth soft-min (low beta) to find a good basin
    try:
        result1 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(20.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-6}
        )
        bary_current = clamp_bary(result1.x.reshape(N, 2))
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.25:
        return bary_current

    # Stage 2: Medium soft-min
    try:
        result2 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(100.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        bary_current = clamp_bary(result2.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.45:
        return bary_current

    # Stage 3: Sharp soft-min (high beta)
    try:
        result3 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(500.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        bary_current = clamp_bary(result3.x.reshape(N, 2))
    except Exception:
        pass

    # Stage 4: Epigraph formulation
    if time.time() - start_time < time_limit * 0.7:
        try:
            # Initialize t with current min area
            current_min = get_min_area(bary_current)
            t_init = current_min * 0.5
            epigraph_params = np.concatenate([bary_current.flatten(), [t_init]])
            
            epigraph_bounds = bounds + [(0, 0.05)]
            
            result_ep = minimize(
                epigraph_obj,
                epigraph_params,
                method='L-BFGS-B',
                bounds=epigraph_bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            # Extract bary from result
            new_bary = clamp_bary(result_ep.x[:2*N].reshape(N, 2))
            
            # Check if this improved the solution
            if get_min_area(new_bary) > get_min_area(bary_current):
                bary_current = new_bary
        except Exception:
            pass
    
    # Stage 5: Final Nelder-Mead polish
    if time.time() - start_time < time_limit * 0.95:
        try:
            result_nm = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-15, 'adaptive': True}
            )
            bary_current = clamp_bary(result_nm.x.reshape(N, 2))
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
        val = get_min_area(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Try Hexagonal Init
    hex_init = generate_hexagonal_init()
    refined_hex = refine_local(hex_init, time_limit=8.0)
    update_best(refined_hex)
    
    # 2. Global Search with Differential Evolution
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
                maxiter=25,
                popsize=12,
                tol=1e-6,
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
    while time.time() < deadline - 12:
        restart_count += 1
        if best_bary is not None and restart_count % 2 == 0:
            # Perturbation of best solution
            init = generate_perturbed_init(best_bary, sigma=0.03, seed=2000 + restart_count)
        else:
            # Random restart
            init = generate_random_init(seed=3000 + restart_count)
        
        remaining = deadline - 12 - time.time()
        if remaining < 4:
            break
            
        t_limit = min(6.0, remaining)
        refined = refine_local(init, time_limit=t_limit)
        update_best(refined)

    # Final high-precision polish on the best candidate
    if best_bary is not None and time.time() < deadline - 3:
        try:
            # Nelder-Mead with high iterations for final polish
            result_nm = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-16, 'adaptive': True}
            )
            update_best(clamp_bary(result_nm.x.reshape(N, 2)))
        except Exception:
            pass
        
        if best_bary is not None and time.time() < deadline - 1:
            try:
                # SLSQP with tight tolerance
                bounds = [(0, 1)] * (N * 2)
                result_slsqp = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-14}
                )
                update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
