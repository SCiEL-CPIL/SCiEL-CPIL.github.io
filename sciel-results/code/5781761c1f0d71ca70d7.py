import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import combinations
import time

N = 11
TIME_LIMIT = 148.0  # Safety margin for 150s limit

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Pre-compute which triangles involve each point
point_tris = [[] for _ in range(N)]
for i, (a, b, c) in enumerate(tri_indices):
    point_tris[a].append(i)
    point_tris[b].append(i)
    point_tris[c].append(i)
point_tris = [np.array(t) for t in point_tris]

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

def calc_areas_for_point(bary, p_idx):
    """Calculate areas only for triangles involving point p_idx."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    tri_set = point_tris[p_idx]
    if len(tri_set) == 0:
        return np.array([])
    
    ui = u[tri_idx[tri_set, 0]]
    vi = v[tri_idx[tri_set, 0]]
    wi = w[tri_idx[tri_set, 0]]
    
    uj = u[tri_idx[tri_set, 1]]
    vj = v[tri_idx[tri_set, 1]]
    wj = w[tri_idx[tri_set, 1]]
    
    uk = u[tri_idx[tri_set, 2]]
    vk = v[tri_idx[tri_set, 2]]
    wk = w[tri_idx[tri_set, 2]]
    
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
    Uses shifted areas for numerical stability.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    if np.any(areas < 1e-15):
        return 1e10
    
    # Shift areas so min is at 0 for numerical stability
    min_area = np.min(areas)
    shifted = areas - min_area
    
    scaled = -beta * shifted
    max_scaled = np.max(scaled)
    
    if max_scaled > 700:
        return -min_area
        
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = min_area - log_sum / beta
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
        penalty = np.sum(np.maximum(0, -diff) ** 2) * 1e6
        return -t + penalty
    
    # Soft constraint via log-sum-exp
    if np.any(areas < 1e-15):
        return 1e10
    
    # Use shifted diff for stability
    min_diff = np.min(diff)
    shifted_diff = diff - min_diff
    
    scaled = -beta * shifted_diff
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        return -t + 1e6 * np.sum(np.maximum(0, -diff) ** 2)
        
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    penalty = max(0, log_sum / beta) * 1e4
    return -t + penalty

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration with more variants.
    """
    best_bary = None
    best_val = -np.inf
    
    # More scale values and offsets
    for scale in [0.12, 0.15, 0.18, 0.2, 0.22, 0.25, 0.28, 0.3, 0.35, 0.4, 0.45, 0.5]:
        for offset in [0.0, 0.5]:
            pts = []
            max_k = int(1.0 / scale) + 1
            for k in range(max_k + 1):
                w_val = k * scale
                if w_val > 1.0:
                    break
                num_pts_in_row = int((1 - w_val) / scale) + 1
                for j in range(num_pts_in_row):
                    u_val = j * scale + offset * scale
                    v_val = (1 - w_val) - u_val
                    if u_val >= -1e-9 and v_val >= -1e-9 and u_val + v_val <= 1 + 1e-9 and u_val <= 1 and v_val <= 1:
                        pts.append([u_val, v_val])
            
            if len(pts) >= N:
                # Try different subsets
                rng = np.random.default_rng(42 + int(scale * 100) + int(offset * 10))
                for _ in range(8):
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

def coordinate_descent(bary_init, time_limit=2.0):
    """
    Efficient coordinate descent for max-min triangle area.
    Uses only relevant triangles for each point evaluation.
    """
    start_time = time.time()
    bary = bary_init.copy()
    
    def eval_point(bary, p_idx, u_val, v_val):
        """Evaluate min area for point p_idx at (u_val, v_val) using only relevant triangles."""
        old_u, old_v = bary[p_idx]
        bary[p_idx, 0] = u_val
        bary[p_idx, 1] = v_val
        areas = calc_areas_for_point(bary, p_idx)
        val = np.min(areas) if len(areas) > 0 else 0.0
        bary[p_idx, 0] = old_u
        bary[p_idx, 1] = old_v
        return val
    
    # Iterate: for each point, do ternary search in u and v
    for sweep in range(50):
        if time.time() - start_time > time_limit:
            break
        improved = False
        for p_idx in range(N):
            u0, v0 = bary[p_idx]
            # Ternary search in u (v fixed)
            lo_u, hi_u = max(1e-9, u0 - 0.15), min(1 - 1e-9, u0 + 0.15)
            # Ensure valid: u + v <= 1 - eps
            hi_u = min(hi_u, 1 - v0 - 1e-9)
            if hi_u - lo_u > 1e-10:
                for _ in range(30):
                    m1 = lo_u + (hi_u - lo_u) / 3
                    m2 = hi_u - (hi_u - lo_u) / 3
                    f1 = eval_point(bary, p_idx, m1, v0)
                    f2 = eval_point(bary, p_idx, m2, v0)
                    if f1 < f2:
                        hi_u = m2
                    else:
                        lo_u = m1
                best_u = (lo_u + hi_u) / 2
                # Check improvement
                old_val = eval_point(bary, p_idx, u0, v0)
                new_val = eval_point(bary, p_idx, best_u, v0)
                if new_val > old_val + 1e-12:
                    bary[p_idx, 0] = best_u
                    improved = True
            # Ternary search in v (u fixed)
            u_now = bary[p_idx, 0]
            lo_v, hi_v = max(1e-9, v0 - 0.15), min(1 - 1e-9, v0 + 0.15)
            hi_v = min(hi_v, 1 - u_now - 1e-9)
            if hi_v - lo_v > 1e-10:
                for _ in range(30):
                    m1 = lo_v + (hi_v - lo_v) / 3
                    m2 = hi_v - (hi_v - lo_v) / 3
                    f1 = eval_point(bary, p_idx, u_now, m1)
                    f2 = eval_point(bary, p_idx, u_now, m2)
                    if f1 < f2:
                        hi_v = m2
                    else:
                        lo_v = m1
                best_v = (lo_v + hi_v) / 2
                old_val = eval_point(bary, p_idx, u_now, v0)
                new_val = eval_point(bary, p_idx, u_now, best_v)
                if new_val > old_val + 1e-12:
                    bary[p_idx, 1] = best_v
                    improved = True
        if not improved:
            break
    
    return clamp_bary(bary)

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages with graduated soft-min.
    Improved time allocation and more aggressive polish.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Coarse soft-min (low beta)
    try:
        result1 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(15.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result1.success or result1.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result1.x.reshape(N, 2))
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.20:
        return bary_current

    # Stage 2: Medium soft-min
    try:
        result2 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(40.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-9}
        )
        if result2.success or result2.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result2.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.40:
        return bary_current

    # Stage 3: Sharp soft-min
    try:
        result3 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(150.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if result3.success or result3.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result3.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.55:
        return bary_current

    # Stage 4: Coordinate descent
    try:
        bary_cd = coordinate_descent(bary_current, time_limit=min(3.0, time_limit * 0.20))
        val_cd = -hard_min_obj(bary_cd.flatten())
        val_cur = -hard_min_obj(bary_current.flatten())
        if val_cd > val_cur:
            bary_current = bary_cd
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.70:
        return bary_current

    # Stage 5: Hard min refinement with Nelder-Mead
    try:
        result4 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
        )
        bary_current = clamp_bary(result4.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.85:
        return bary_current

    # Stage 6: Final high-precision Nelder-Mead polish
    try:
        result5 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1500, 'xatol': 1e-15, 'fatol': 1e-17, 'adaptive': True}
        )
        bary_current = clamp_bary(result5.x.reshape(N, 2))
    except Exception:
        pass
        
    # Stage 7: Final coordinate descent polish
    try:
        bary_cd2 = coordinate_descent(bary_current, time_limit=min(1.5, time_limit * 0.12))
        val_cd2 = -hard_min_obj(bary_cd2.flatten())
        val_cur2 = -hard_min_obj(bary_current.flatten())
        if val_cd2 > val_cur2:
            bary_current = bary_cd2
    except Exception:
        pass
        
    return bary_current

def refine_epigraph(bary_init, time_limit=5.0):
    """
    Refine using epigraph formulation with auxiliary variable t.
    """
    start_time = time.time()
    bary_current = bary_init.copy()
    
    # Estimate initial t from current min area
    areas = calc_areas_bary(bary_current)
    t0 = np.min(areas) * 0.99
    
    # Epigraph optimization
    x0 = np.concatenate([bary_current.flatten(), [t0]])
    bounds = [(0, 1)] * (N * 2) + [(0, 0.5)]
    
    # Graduated beta for epigraph
    for beta in [40.0, 100.0, 250.0]:
        if time.time() - start_time > time_limit * 0.65:
            break
        try:
            result = minimize(
                epigraph_obj,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            new_bary = clamp_bary(result.x[:2*N].reshape(N, 2))
            new_val = -hard_min_obj(new_bary.flatten())
            cur_val = -hard_min_obj(bary_current.flatten())
            if new_val > cur_val:
                bary_current = new_bary
            x0 = np.concatenate([bary_current.flatten(), [np.min(calc_areas_bary(bary_current)) * 0.995]])
        except Exception:
            pass
    
    # Final NM polish
    if time.time() - start_time < time_limit * 0.9:
        try:
            result_nm = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2500, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
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
        val = -hard_min_obj(bary.flatten())
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Try Hexagonal Init with standard refinement
    hex_init = generate_hexagonal_init()
    refined_hex = refine_local(hex_init, time_limit=10.0)
    update_best(refined_hex)

    # 2. Try epigraph refinement on the same hex init
    if time.time() < deadline - 45:
        refined_ep = refine_epigraph(hex_init, time_limit=8.0)
        update_best(refined_ep)

    # 3. Global Search with Differential Evolution (increased population)
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
                maxiter=60,
                popsize=20,
                tol=1e-8,
                polish=False
            )

            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined_de = refine_local(de_bary, time_limit=10.0)
            update_best(refined_de)
        except Exception:
            pass

    # 4. Multi-start: random + perturbed restarts with local refinement
    restart_count = 0
    while time.time() < deadline - 20:
        restart_count += 1
        if best_bary is not None and restart_count % 3 == 0:
            # Larger perturbation of best solution
            init = generate_perturbed_init(best_bary, sigma=0.08, seed=2000 + restart_count)
        elif best_bary is not None and restart_count % 3 == 1:
            # Smaller perturbation
            init = generate_perturbed_init(best_bary, sigma=0.03, seed=2000 + restart_count)
        else:
            # Fresh random start
            init = generate_random_init(seed=3000 + restart_count)
        
        refined = refine_local(init, time_limit=4.0)
        update_best(refined)

    # 5. Final high-precision polish
    if best_bary is not None and time.time() < deadline - 10:
        # Try epigraph on best with higher beta values
        refined_final_ep = refine_epigraph(best_bary, time_limit=6.0)
        update_best(refined_final_ep)
        
        if best_bary is not None and time.time() < deadline - 6:
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
        
        if best_bary is not None and time.time() < deadline - 4:
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 6000, 'xatol': 1e-15, 'fatol': 1e-18, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass
        
        # Coordinate descent polish on best
        if best_bary is not None and time.time() < deadline - 3:
            try:
                bary_cd = coordinate_descent(best_bary, time_limit=min(3.0, deadline - time.time() - 1))
                update_best(bary_cd)
            except Exception:
                pass
        
        # Final epigraph pass with very high beta for precise area balancing
        if best_bary is not None and time.time() < deadline - 2:
            try:
                # Custom high-precision epigraph
                def high_precision_epigraph(x):
                    bary = x[:2*N].reshape(N, 2)
                    t = x[2*N]
                    bary = clamp_bary(bary)
                    areas = calc_areas_bary(bary)
                    
                    diff = areas - t
                    if np.any(diff < -1e-12):
                        penalty = np.sum(np.maximum(0, -diff) ** 2) * 1e8
                        return -t + penalty
                    
                    if np.any(areas < 1e-15):
                        return 1e10
                    
                    # Very sharp soft constraint with shifted values
                    min_diff = np.min(diff)
                    shifted_diff = diff - min_diff
                    beta_hp = 500.0
                    scaled = -beta_hp * shifted_diff
                    max_scaled = np.max(scaled)
                    if max_scaled > 700:
                        return -t + 1e8 * np.sum(np.maximum(0, -diff) ** 2)
                        
                    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
                    penalty = max(0, log_sum / beta_hp) * 1e6
                    return -t + penalty

                areas_cur = calc_areas_bary(best_bary)
                t0 = np.min(areas_cur) * 0.999
                x0 = np.concatenate([best_bary.flatten(), [t0]])
                bounds_ep = [(0, 1)] * (N * 2) + [(0, 0.5)]
                
                result_hp = minimize(
                    high_precision_epigraph,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds_ep,
                    options={'maxiter': 600, 'ftol': 1e-16, 'gtol': 1e-12}
                )
                update_best(clamp_bary(result_hp.x[:2*N].reshape(N, 2)))
            except Exception:
                pass
        
        # One more SLSQP pass
        if best_bary is not None and time.time() < deadline - 1:
            try:
                bounds = [(0, 1)] * (N * 2)
                result_final = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 4000, 'ftol': 1e-16}
                )
                update_best(clamp_bary(result_final.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
