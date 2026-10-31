import numpy as np
from scipy.optimize import minimize
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

def clamp_bary(bary, eps=1e-10):
    """Clamp barycentric coordinates to the strictly valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    u = np.clip(u, eps, 1.0 - eps)
    v = np.clip(v, eps, 1.0 - eps)
    
    sums = u + v
    invalid = sums > (1.0 - eps)
    if np.any(invalid):
        scale = (1.0 - eps) / np.maximum(sums, eps)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    u = np.clip(u, eps, 1.0 - eps)
    v = np.clip(v, eps, 1.0 - eps)
    
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

def force_directed_relax(bary_init, time_limit=10.0, rng=None):
    """
    Force-directed relaxation: treat points as particles with repulsive forces
    weighted by inverse triangle areas. Points in small-area triangles feel
    stronger repulsion, driving the system toward max-min configuration.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    bary = bary_init.copy()
    start_time = time.time()
    
    # Adaptive step size
    step = 0.02
    min_step = 1e-8
    max_step = 0.05
    
    # Precompute which triangles each point participates in
    point_tri_mask = []
    for i in range(N):
        mask = np.zeros(len(tri_idx), dtype=bool)
        mask[np.any(tri_idx == i, axis=1)] = True
        point_tri_mask.append(mask)
    
    prev_val = -hard_min_obj(bary.flatten())
    stall_count = 0
    
    while time.time() - start_time < time_limit:
        areas = calc_areas_bary(bary)
        min_area = np.min(areas)
        
        if min_area < 1e-12:
            # Degenerate: add small random perturbation
            bary = clamp_bary(bary + rng.normal(0, 0.01, bary.shape))
            continue
        
        # Compute force on each point
        forces = np.zeros((N, 2))
        
        for tri_i in range(len(tri_idx)):
            i, j, k = tri_idx[tri_i]
            area = areas[tri_i]
            
            if area < 1e-12:
                continue
            
            # Weight: stronger repulsion for smaller areas
            weight = 1.0 / (area * area + 1e-10)
            
            # Barycentric coordinates of the three points
            u_i, v_i = bary[i, 0], bary[i, 1]
            u_j, v_j = bary[j, 0], bary[j, 1]
            u_k, v_k = bary[k, 0], bary[k, 1]
            
            # Force direction: push points apart along the "altitude" directions
            # In barycentric coords, the gradient of area w.r.t. point i is
            # related to the vector from i to the opposite side
            
            # Gradient of |det| w.r.t. (u_i, v_i):
            # det = u_i*(v_j*w_k - v_k*w_j) - v_i*(u_j*w_k - u_k*w_j) + w_i*(u_j*v_k - u_k*v_j)
            # where w_i = 1 - u_i - v_i
            # d(det)/du_i = (v_j*w_k - v_k*w_j) - (u_j*w_k - u_k*w_j) - (u_j*v_k - u_k*v_j)
            # d(det)/dv_i = (v_j*w_k - v_k*w_j) - (u_j*w_k - u_k*w_j) - (u_j*v_k - u_k*v_j)
            # Wait, let me recompute properly.
            
            w_i = 1.0 - u_i - v_i
            w_j = 1.0 - u_j - v_j
            w_k = 1.0 - u_k - v_k
            
            # det = u_i*(v_j*w_k - v_k*w_j) - v_i*(u_j*w_k - u_k*w_j) + w_i*(u_j*v_k - u_k*v_j)
            # Since w_i = 1 - u_i - v_i:
            # det = u_i*(v_j*w_k - v_k*w_j - u_j*v_k + u_k*v_j) - v_i*(u_j*w_k - u_k*w_j - u_j*v_k + u_k*v_j) + (1-u_i-v_i)*(u_j*v_k - u_k*v_j)
            
            A = v_j*w_k - v_k*w_j
            B = u_j*w_k - u_k*w_j
            C = u_j*v_k - u_k*v_j
            
            # det = u_i*A - v_i*B + (1-u_i-v_i)*C = u_i*(A-C) - v_i*(B+C) + C
            
            grad_u = A - C
            grad_v = -(B + C)
            
            # Force on point i: proportional to gradient of area, pointing away
            # We want to INCREASE the area, so force is in direction of gradient
            f_i = weight * np.array([grad_u, grad_v])
            
            # For point j:
            # By symmetry, gradient w.r.t. (u_j, v_j)
            A_j = v_k*w_i - v_i*w_k
            B_j = u_k*w_i - u_i*w_k
            C_j = u_k*v_i - u_i*v_k
            grad_u_j = A_j - C_j
            grad_v_j = -(B_j + C_j)
            f_j = weight * np.array([grad_u_j, grad_v_j])
            
            # For point k:
            A_k = v_i*w_j - v_j*w_i
            B_k = u_i*w_j - u_j*w_i
            C_k = u_i*v_j - u_j*v_i
            grad_u_k = A_k - C_k
            grad_v_k = -(B_k + C_k)
            f_k = weight * np.array([grad_u_k, grad_v_k])
            
            forces[i] += f_i
            forces[j] += f_j
            forces[k] += f_k
        
        # Normalize forces to prevent explosion
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms = np.maximum(force_norms, 1e-12)
        forces_normed = forces / force_norms
        
        # Update positions
        bary_new = bary + step * forces_normed
        bary_new = clamp_bary(bary_new)
        
        # Check improvement
        new_val = -hard_min_obj(bary_new.flatten())
        
        if new_val > prev_val:
            bary = bary_new
            prev_val = new_val
            stall_count = 0
            # Slowly decrease step for convergence
            step = max(min_step, step * 0.995)
        else:
            # Rejected: try smaller step
            step = max(min_step, step * 0.5)
            stall_count += 1
            
            if stall_count > 50:
                # Add random perturbation to escape local minimum
                bary = clamp_bary(bary + rng.normal(0, 0.02, bary.shape))
                step = 0.02
                stall_count = 0
    
    return bary

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using force-directed relaxation followed by
    gradient-based polish.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Force-directed relaxation
    if time.time() - start_time < time_limit * 0.5:
        fd_time = min(time_limit * 0.45, 5.0)
        bary_fd = force_directed_relax(bary_current, time_limit=fd_time, rng=np.random.default_rng(42))
        val_fd = -hard_min_obj(bary_fd.flatten())
        val_curr = -hard_min_obj(bary_current.flatten())
        if val_fd > val_curr:
            bary_current = bary_fd
    
    if time.time() - start_time > time_limit * 0.6:
        return bary_current
    
    # Stage 2: Soft-min gradient optimization
    try:
        result1 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(50.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-9}
        )
        val1 = -result1.fun
        val_curr = -hard_min_obj(bary_current.flatten())
        if val1 > val_curr:
            bary_current = clamp_bary(result1.x.reshape(N, 2))
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.8:
        return bary_current
    
    # Stage 3: Hard min with SLSQP
    try:
        result2 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14}
        )
        bary_current = clamp_bary(result2.x.reshape(N, 2))
    except Exception:
        pass
    
    return bary_current

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

    # 1. Try Hexagonal Init with force-directed relaxation
    hex_init = generate_hexagonal_init()
    refined_hex = refine_local(hex_init, time_limit=10.0)
    update_best(refined_hex)
    
    # 2. Multiple force-directed relaxations from diverse starts
    rng = np.random.default_rng(123)
    for attempt in range(5):
        if time.time() > deadline - 20:
            break
        
        if attempt == 0:
            init = generate_random_init(seed=100)
        elif attempt == 1:
            init = generate_random_init(seed=200)
        elif attempt == 2:
            init = generate_perturbed_init(hex_init, sigma=0.1, seed=300)
        elif attempt == 3:
            init = generate_perturbed_init(best_bary, sigma=0.08, seed=400)
        else:
            init = generate_random_init(seed=500 + attempt)
        
        # Pure force-directed relaxation with more time
        fd_time = min(8.0, deadline - time.time() - 15)
        if fd_time > 2.0:
            bary_fd = force_directed_relax(init, time_limit=fd_time, rng=rng)
            update_best(bary_fd)
    
    # 3. Final polish on best solution
    if best_bary is not None and time.time() < deadline - 8:
        refined = refine_local(best_bary, time_limit=min(6.0, deadline - time.time() - 3))
        update_best(refined)
    
    # 4. Ultra-fine SLSQP polish
    if best_bary is not None and time.time() < deadline - 3:
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
    
    # 5. Nelder-Mead final
    if best_bary is not None and time.time() < deadline - 1:
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

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
