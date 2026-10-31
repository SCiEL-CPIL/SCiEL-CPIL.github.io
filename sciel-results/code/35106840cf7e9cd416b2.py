import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 90.0  # Safe margin for 150s limit

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
            for _ in range(3):
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

def generate_diverse_inits():
    """
    Generate a set of diverse initial configurations.
    """
    inits = []
    # Hexagonal
    inits.append(generate_hexagonal_init())
    
    # Random Uniform
    rng = np.random.default_rng(123)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    inits.append(np.column_stack([u, v]))
    
    # Corner-biased (points pushed towards vertices)
    rng = np.random.default_rng(456)
    u = rng.random(N) ** 2
    v = rng.random(N) ** 2 * (1 - u)
    inits.append(np.column_stack([u, v]))
    
    # Center-biased
    rng = np.random.default_rng(789)
    u = 0.5 + (rng.random(N) - 0.5) * 0.5
    v = 0.5 + (rng.random(N) - 0.5) * 0.5 * (1 - u)
    inits.append(np.column_stack([u, v]))
    
    return inits

def generate_random_init(rng_seed=42):
    """Generate a random initial configuration."""
    rng = np.random.default_rng(rng_seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def generate_perturbed_init(bary, scale=0.05, rng_seed=42):
    """Generate a perturbed version of an existing configuration."""
    rng = np.random.default_rng(rng_seed)
    noise = rng.normal(0, scale, bary.shape)
    return clamp_bary(bary + noise)

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Smooth soft-min minimization (L-BFGS-B is fast)
    try:
        if time.time() - start_time < time_limit * 0.3:
            result1 = minimize(
                soft_min_obj,
                bary_current.flatten(),
                args=(50.0,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result1.success or result1.fun < -hard_min_obj(bary_current.flatten()):
                bary_current = clamp_bary(result1.x.reshape(N, 2))
    except Exception:
        pass
    
    # Stage 2: Hard min refinement with Nelder-Mead
    try:
        if time.time() - start_time < time_limit * 0.7:
            result2 = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-14, 'adaptive': True}
            )
            bary_current = clamp_bary(result2.x.reshape(N, 2))
    except Exception:
        pass
        
    # Stage 3: SLSQP polish (optional if time permits)
    try:
        if time.time() - start_time < time_limit * 0.9:
            result3 = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12}
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

    # 1. Multi-Start Local Refinement
    inits = generate_diverse_inits()
    
    # Allocate time per start
    num_starts = len(inits)
    time_per_start = (TIME_LIMIT - 10.0) / num_starts # Reserve 10s for final polish
    
    for i, init in enumerate(inits):
        if time.time() > deadline - 10.0:
            break
            
        # Dynamic time limit for refinement based on remaining time
        remaining = deadline - 10.0 - time.time()
        t_limit = min(time_per_start, remaining)
        
        if t_limit < 1.0:
            continue
            
        refined = refine_local(init, time_limit=t_limit)
        update_best(refined)
        
        # If we have time, do a few more random restarts from the best found
        if best_bary is not None and time.time() < deadline - 20.0:
             # Generate a few perturbations of the best
             rng = np.random.default_rng(i)
             for _ in range(2):
                 if time.time() > deadline - 15.0:
                     break
                 noise = rng.normal(0, 0.05, best_bary.shape)
                 perturbed = clamp_bary(best_bary + noise)
                 refined_p = refine_local(perturbed, time_limit=2.0)
                 update_best(refined_p)

    # 2. Final Polish
    if best_bary is not None and time.time() < deadline - 2.0:
        try:
            bounds = [(0, 1)] * (N * 2)
            # Use Nelder-Mead for final polish as it handles non-smoothness better
            result_final = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
            )
            update_best(clamp_bary(result_final.x.reshape(N, 2)))
        except Exception:
            pass

    # Convert to cartesian
    if best_bary is None:
        # Fallback
        best_bary = generate_hexagonal_init()
        
    cart = bary_to_cart(best_bary)
    
    return cart
