import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

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

def clamp_bary(bary, eps=1e-12):
    """Clamp barycentric coordinates to the strictly valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    u = np.clip(u, eps, 1.0 - eps)
    v = np.clip(v, eps, 1.0 - eps)
    
    sums = u + v
    invalid = sums > (1.0 - eps)
    if np.any(invalid):
        scale = (1.0 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    u = np.clip(u, eps, 1.0 - eps)
    v = np.clip(v, eps, 1.0 - eps)
    
    return np.column_stack([u, v])

def min_area_analytical_grad(bary, n_active=3):
    """
    Compute the minimum area and its analytical gradient with respect to barycentric coordinates.
    Returns (min_area, grad) where grad has shape (N, 2).
    Uses the sign of the determinant to handle the absolute value correctly.
    Averages gradient over the n_active smallest triangles to handle near-degenerate cases.
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # Compute all determinants (signed)
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
    areas = np.abs(det)
    
    # Find the n_active smallest triangles
    min_area = np.min(areas)
    
    # Get indices of triangles with area close to minimum
    threshold = min_area + 1e-10
    active_mask = areas <= threshold
    active_indices = np.where(active_mask)[0]
    
    # Limit to n_active if too many
    if len(active_indices) > n_active:
        # Take the n_active smallest
        sorted_active = active_indices[np.argsort(areas[active_indices])][:n_active]
        active_indices = sorted_active
    
    grad = np.zeros_like(bary)
    
    for min_idx in active_indices:
        sign = np.sign(det[min_idx])
        if sign == 0:
            sign = 1.0
        
        # For the minimum triangle (i, j, k), compute gradient of sign * det
        i, j, k = tri_idx[min_idx]
        
        ui, vi, wi = u[i], v[i], w[i]
        uj, vj, wj = u[j], v[j], w[j]
        uk, vk, wk = u[k], v[k], w[k]
        
        # Cofactor-based gradient computation
        C11 = vj * wk - vk * wj
        C12 = uk * wj - uj * wk
        C13 = uj * vk - uk * vj
        
        d_det_d_ui = C11 - C13
        d_det_d_vi = C12 - C13
        
        C21 = vk * wi - vi * wk
        C22 = ui * wk - uk * wi
        C23 = uj * vi - ui * vj
        
        d_det_d_uj = C21 - C23
        d_det_d_vj = C22 - C23
        
        C31 = vi * wj - vj * wi
        C32 = uj * wi - ui * wj
        C33 = ui * vj - uj * vi
        
        d_det_d_uk = C31 - C33
        d_det_d_vk = C32 - C33
        
        # Accumulate gradient (for sign * det)
        grad[i, 0] += sign * d_det_d_ui
        grad[i, 1] += sign * d_det_d_vi
        grad[j, 0] += sign * d_det_d_uj
        grad[j, 1] += sign * d_det_d_vj
        grad[k, 0] += sign * d_det_d_uk
        grad[k, 1] += sign * d_det_d_vk
    
    # Normalize by number of active constraints
    grad /= len(active_indices)
    
    # We want to maximize min_area, so minimize -min_area
    # Gradient of -min_area = -gradient of min_area
    return min_area, -grad

def objective_and_grad(bary_flat):
    """Objective function with analytical gradient for SLSQP."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    min_area, grad = min_area_analytical_grad(bary)
    return -min_area, grad

def hard_min_obj(bary_flat):
    """Objective using hard min."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def generate_structured_init():
    """
    Generate a structured initial configuration using hexagonal packing ideas.
    """
    best_bary = None
    best_val = -np.inf
    
    # Try multiple structured patterns
    patterns = [
        # Hexagonal-like pattern
        lambda: _hex_pattern(42),
        lambda: _hex_pattern(123),
        lambda: _grid_pattern(3, 4, 42),
        lambda: _grid_pattern(4, 3, 42),
        lambda: _concentric_pattern(42),
    ]
    
    for gen_func in patterns:
        bary = gen_func()
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

def _hex_pattern(seed):
    """Generate a hexagonal-like pattern."""
    rng = np.random.default_rng(seed)
    # Start with a base hexagonal arrangement and perturb
    base = np.array([
        [0.5, 0.5],
        [0.25, 0.25],
        [0.75, 0.25],
        [0.25, 0.75],
        [0.75, 0.75],
        [0.5, 0.1],
        [0.5, 0.9],
        [0.1, 0.5],
        [0.9, 0.5],
        [0.35, 0.35],
        [0.65, 0.65],
    ])
    # Perturb slightly
    bary = base + rng.normal(0, 0.02, base.shape)
    return bary

def _grid_pattern(n_rows, n_cols, seed):
    """Generate a grid pattern."""
    rng = np.random.default_rng(seed)
    pts = []
    for r in range(n_rows + 1):
        w_val = r / n_rows
        if w_val > 1.0:
            continue
        u_max = 1.0 - w_val
        for c in range(n_cols + 1):
            u_val = c * u_max / n_cols
            v_val = u_max - u_val
            if u_val >= -1e-9 and v_val >= -1e-9 and u_val + v_val <= 1 + 1e-9:
                pts.append([u_val, v_val])
    
    if len(pts) >= N:
        if len(pts) == N:
            bary = np.array(pts)
        else:
            idx = rng.choice(len(pts), N, replace=False)
            bary = np.array(pts)[idx]
    else:
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        bary = np.column_stack([u, v])
    
    return bary

def _concentric_pattern(seed):
    """Generate concentric ring pattern."""
    rng = np.random.default_rng(seed)
    center = np.array([0.5, 0.5])
    bary = np.zeros((N, 2))
    
    # Place 1 at center, 6 in inner ring, 4 in outer ring
    bary[0] = center
    for i in range(1, 7):
        angle = 2 * np.pi * (i - 1) / 6
        r = 0.2
        bary[i] = center + r * np.array([np.cos(angle), np.sin(angle)])
    for i in range(7, 11):
        angle = 2 * np.pi * (i - 7) / 4 + np.pi / 4
        r = 0.35
        bary[i] = center + r * np.array([np.cos(angle), np.sin(angle)])
    
    bary = bary + rng.normal(0, 0.01, bary.shape)
    return bary

def refine_analytical(bary_init, time_limit=10.0):
    """
    Refine using gradient-based SLSQP with analytical gradient.
    """
    start_time = time.time()
    bary_current = bary_init.copy()
    
    bounds = [(0, 1)] * (N * 2)
    
    # Stage 1: SLSQP with analytical gradient
    try:
        result = minimize(
            lambda x: objective_and_grad(x)[0],
            bary_current.flatten(),
            jac=lambda x: objective_and_grad(x)[1],
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-12, 'disp': False}
        )
        
        if result.fun < hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result.x.reshape(N, 2))
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.5:
        return bary_current
    
    # Stage 2: Another SLSQP pass with tighter tolerance
    try:
        result2 = minimize(
            lambda x: objective_and_grad(x)[0],
            bary_current.flatten(),
            jac=lambda x: objective_and_grad(x)[1],
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14, 'disp': False}
        )
        bary_current = clamp_bary(result2.x.reshape(N, 2))
    except Exception:
        pass
    
    return bary_current

def local_search_barycentric(bary_init, time_limit=5.0):
    """
    Fast local search using coordinate descent on barycentric coords.
    Optimized to avoid unnecessary array copies.
    """
    start_time = time.time()
    bary = bary_init.copy()
    best_bary = bary.copy()
    best_val = -hard_min_obj(bary.flatten())
    
    step_size = 0.05
    improvement_threshold = 1e-12
    
    while time.time() - start_time < time_limit:
        improved = False
        for _ in range(200):  # Max iterations per step size
            if time.time() - start_time > time_limit:
                break
            
            # Try moving each point in each coordinate
            for p in range(N):
                for c in range(2):
                    # Try positive and negative moves
                    for delta in [step_size, -step_size]:
                        old_val = bary[p, c]
                        bary[p, c] = old_val + delta
                        bary = clamp_bary(bary)
                        val = -hard_min_obj(bary.flatten())
                        
                        if val > best_val + improvement_threshold:
                            best_val = val
                            best_bary = bary.copy()
                            improved = True
                        else:
                            bary[p, c] = old_val
                            bary = clamp_bary(bary)
                        
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        if not improved:
            step_size *= 0.5
            if step_size < 1e-7:
                break
    
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

    # 1. Structured initialization
    struct_init = generate_structured_init()
    refined_struct = refine_analytical(struct_init, time_limit=8.0)
    update_best(refined_struct)
    
    # 2. Local search from best
    if best_bary is not None and time.time() < deadline - 30:
        ls_result = local_search_barycentric(best_bary, time_limit=10.0)
        update_best(ls_result)
        
        # Refine after local search
        if time.time() < deadline - 15:
            refined_ls = refine_analytical(ls_result, time_limit=8.0)
            update_best(refined_ls)
    
    # 3. Multi-start with analytical gradient refinement
    rng = np.random.default_rng(123)
    restart_count = 0
    while time.time() < deadline - 20:
        restart_count += 1
        if best_bary is not None and restart_count % 3 == 0:
            # Perturbation of best with adaptive sigma
            sigma = 0.03 + 0.02 * (restart_count // 10)
            init = best_bary + rng.normal(0, sigma, best_bary.shape)
        else:
            # Fresh random with slight bias toward center
            u = rng.random(N)
            v = rng.random(N) * (1 - u)
            init = np.column_stack([u, v])
        
        init = clamp_bary(init)
        refined = refine_analytical(init, time_limit=5.0)
        update_best(refined)
        
        # Occasionally do local search
        if time.time() < deadline - 15 and restart_count % 4 == 0:
            ls_short = local_search_barycentric(refined, time_limit=3.0)
            update_best(ls_short)
    
    # 4. Final high-precision gradient polish
    if best_bary is not None and time.time() < deadline - 5:
        try:
            result_final = minimize(
                lambda x: objective_and_grad(x)[0],
                best_bary.flatten(),
                jac=lambda x: objective_and_grad(x)[1],
                method='SLSQP',
                bounds=[(0, 1)] * (N * 2),
                options={'maxiter': 1000, 'ftol': 1e-16, 'disp': False}
            )
            update_best(clamp_bary(result_final.x.reshape(N, 2)))
        except Exception:
            pass
        
        # One more pass
        if best_bary is not None and time.time() < deadline - 2:
            try:
                result_final2 = minimize(
                    lambda x: objective_and_grad(x)[0],
                    best_bary.flatten(),
                    jac=lambda x: objective_and_grad(x)[1],
                    method='SLSQP',
                    bounds=[(0, 1)] * (N * 2),
                    options={'maxiter': 2000, 'ftol': 1e-18, 'disp': False}
                )
                update_best(clamp_bary(result_final2.x.reshape(N, 2)))
            except Exception:
                pass
    
    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
