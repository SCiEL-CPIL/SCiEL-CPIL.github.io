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

def hard_min_obj(bary_flat):
    """Objective using hard min."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def epigraph_obj(x, beta=100.0):
    """
    Epigraph formulation: x = [bary_flat (22), t (1)]
    Minimize -t subject to soft constraints: all areas >= t.
    """
    bary = x[:2*N].reshape(N, 2)
    t = x[2*N]
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Penalty for areas < t (hard violation)
    diff = areas - t
    neg_diff = np.maximum(0, -diff)
    if np.any(neg_diff > 1e-10):
        penalty = np.sum(neg_diff ** 2) * 1e6
        return -t + penalty
    
    # Soft constraint via log-sum-exp for near-violations
    if np.any(areas < 1e-15):
        return 1e10
    
    scaled = -beta * diff
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        return -t + 1e6 * np.sum(neg_diff)
    
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_penalty = log_sum / beta
    return -t + soft_penalty

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

def generate_radial_init(seed=None):
    """Generate points arranged in concentric rings around centroid."""
    rng = np.random.default_rng(seed)
    # Centroid of triangle in barycentric: (1/3, 1/3)
    cx, cy = 1.0/3, 1.0/3
    pts = [[cx, cy]]
    
    # Ring 1: 6 points at radius ~0.15
    r1 = 0.15
    for i in range(6):
        angle = i * np.pi / 3
        u = cx + r1 * np.cos(angle)
        v = cy + r1 * np.sin(angle) * 0.866  # Adjust for equilateral geometry
        pts.append([u, v])
    
    # Ring 2: 4 more points at radius ~0.3
    r2 = 0.3
    for i in range(4):
        angle = i * np.pi / 2 + np.pi / 4
        u = cx + r2 * np.cos(angle)
        v = cy + r2 * np.sin(angle) * 0.866
        pts.append([u, v])
    
    bary = np.array(pts[:N])
    return clamp_bary(bary)

def generate_boundary_init(seed=None):
    """Generate points along the boundary and interior."""
    rng = np.random.default_rng(seed)
    pts = []
    
    # 3 corners
    pts.append([0.05, 0.05])
    pts.append([0.95, 0.05])
    pts.append([0.45, 0.5])
    
    # 4 edge midpoints
    pts.append([0.5, 0.05])
    pts.append([0.05, 0.45])
    pts.append([0.5, 0.5])
    pts.append([0.95, 0.45])
    
    # 4 interior points
    for _ in range(4):
        u = rng.uniform(0.2, 0.8)
        v = rng.uniform(0.1, 0.7 * (1 - u))
        pts.append([u, v])
    
    bary = np.array(pts[:N])
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

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages including epigraph.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Smooth soft-min minimization
    try:
        result1 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(50.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if result1.success or result1.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result1.x.reshape(N, 2))
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.3:
        return bary_current
        
    # Stage 2: Epigraph optimization (smooth constrained)
    try:
        t_init = -hard_min_obj(bary_current.flatten())
        x_init = np.concatenate([bary_current.flatten(), [t_init]])
        bounds_epigraph = [(0, 1)] * (N * 2) + [(0, 1)]
        
        result2 = minimize(
            epigraph_obj,
            x_init,
            args=(100.0,),
            method='L-BFGS-B',
            bounds=bounds_epigraph,
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        ep_bary = clamp_bary(result2.x[:2*N].reshape(N, 2))
        ep_val = -hard_min_obj(ep_bary.flatten())
        if ep_val > -hard_min_obj(bary_current.flatten()):
            bary_current = ep_bary
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.6:
        return bary_current
    
    # Stage 3: Hard min refinement with Nelder-Mead
    try:
        result3 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1500, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
        )
        bary_current = clamp_bary(result3.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.85:
        return bary_current
    
    # Stage 4: SLSQP polish
    try:
        result4 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14}
        )
        bary_current = clamp_bary(result4.x.reshape(N, 2))
    except Exception:
        pass
        
    return bary_current

def epigraph_polish(bary_init, time_limit=3.0):
    """
    High-precision epigraph polish with very high beta.
    """
    start_time = time.time()
    
    t_init = -hard_min_obj(bary_init.flatten())
    x_init = np.concatenate([bary_init.flatten(), [t_init]])
    bounds_epigraph = [(0, 1)] * (N * 2) + [(0, 1)]
    
    best_bary = bary_init.copy()
    best_val = t_init
    
    # Try multiple high beta values
    for beta in [500, 1000]:
        if time.time() - start_time > time_limit * 0.5:
            break
        try:
            result = minimize(
                epigraph_obj,
                x_init,
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds_epigraph,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            ep_bary = clamp_bary(result.x[:2*N].reshape(N, 2))
            ep_val = -hard_min_obj(ep_bary.flatten())
            if ep_val > best_val:
                best_val = ep_val
                best_bary = ep_bary
                x_init = np.concatenate([ep_bary.flatten(), [ep_val]])
        except Exception:
            pass
    
    # Final SLSQP on hard min
    if time.time() - start_time < time_limit:
        try:
            bounds = [(0, 1)] * (N * 2)
            result_final = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15}
            )
            final_bary = clamp_bary(result_final.x.reshape(N, 2))
            final_val = -hard_min_obj(final_bary.flatten())
            if final_val > best_val:
                best_bary = final_bary
        except Exception:
            pass
    
    return best_bary

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

    # 1. Try multiple structured inits
    inits = [
        generate_hexagonal_init(),
        generate_radial_init(seed=42),
        generate_boundary_init(seed=42),
    ]
    for init in inits:
        if time.time() < deadline - 40:
            refined = refine_local(init, time_limit=5.0)
            update_best(refined)
    
    # 2. Global Search with Differential Evolution
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
                maxiter=30,
                popsize=12,
                tol=1e-6,
                polish=False
            )
            
            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined_de = refine_local(de_bary, time_limit=6.0)
            update_best(refined_de)
        except Exception:
            pass
    
    # 3. Multi-start: random + perturbed restarts with local refinement
    restart_count = 0
    while time.time() < deadline - 15:
        restart_count += 1
        if best_bary is not None and restart_count % 3 == 0:
            # Perturbation of best solution
            init = generate_perturbed_init(best_bary, sigma=0.03, seed=2000 + restart_count)
        elif restart_count % 3 == 1:
            # Random restart
            init = generate_random_init(seed=3000 + restart_count)
        else:
            # Radial or boundary variant
            if restart_count % 6 == 1:
                init = generate_radial_init(seed=4000 + restart_count)
            else:
                init = generate_boundary_init(seed=5000 + restart_count)
        
        remaining = deadline - 15 - time.time()
        if remaining < 3:
            break
            
        t_limit = min(4.0, remaining)
        refined = refine_local(init, time_limit=t_limit)
        update_best(refined)

    # Final high-precision epigraph polish on the best candidate
    if best_bary is not None and time.time() < deadline - 8:
        polished = epigraph_polish(best_bary, time_limit=6.0)
        update_best(polished)
    
    # Final SLSQP + Nelder-Mead polish
    if best_bary is not None and time.time() < deadline - 3:
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
        
        if best_bary is not None and time.time() < deadline - 1:
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-13, 'fatol': 1e-16, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
