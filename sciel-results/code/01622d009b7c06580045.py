"""
Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle.
Strategy: Multi-start optimization with barycentric coordinates, using a combination of
L-BFGS-B with soft-min objective, Nelder-Mead polish, and simulated annealing.
"""
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11
H = np.sqrt(3) / 2.0

# Pre-compute triangle indices
tri_idx = np.array(list(combinations(range(N), 3)))  # (165, 3)

def bary_to_cart(bary):
    """
    Convert barycentric coords (u, v) to Cartesian (x, y).
    w = 1 - u - v
    Point = u*A + v*B + w*C where A=(0,0), B=(1,0), C=(0.5, H)
    x = v + 0.5*w
    y = w * H
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def calc_areas_bary(bary):
    """
    Calculate areas of all triangles formed by N points given in barycentric coords (u, v).
    Returns array of shape (165,) containing the area ratios (relative to big triangle area).
    """
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

def clamp_bary(bary, eps=1e-8):
    """
    Clamp barycentric coordinates to the valid triangle region.
    u >= eps, v >= eps, u+v <= 1 - eps
    """
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

def hard_min_obj(bary_flat):
    """
    Objective function using hard min. Returns negative min area.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def soft_min_obj(bary_flat, beta=100.0):
    """
    Objective function using soft min. Returns negative soft-min area.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        return -np.min(areas)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    
    return -soft_min

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration in barycentric coords.
    """
    best_bary = None
    best_val = -np.inf
    
    for s in np.linspace(0.08, 0.45, 30):
        points = []
        for i in range(8):
            for j in range(8):
                u = i * s
                v = j * s
                if u + v <= 1.0 and len(points) < N:
                    points.append([u, v])
        
        if len(points) == N:
            bary = np.array(points)
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

def generate_perturbed_init(base, noise_scale=0.05, seed=None):
    """Generate perturbed version of a base configuration."""
    rng = np.random.default_rng(seed)
    bary = base.copy()
    bary += rng.normal(0, noise_scale, bary.shape)
    return clamp_bary(bary)

def generate_structured_init(seed=None):
    """
    Generate a structured initial configuration based on symmetry.
    Place points at vertices, edge midpoints, and centroid-like positions.
    """
    rng = np.random.default_rng(seed)
    # Vertices
    bary = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    # Edge midpoints
    bary = np.vstack([bary, [
        [0.5, 0.0],
        [0.0, 0.5],
        [0.5, 0.5],
    ]])
    # Centroid
    bary = np.vstack([bary, [[1/3, 1/3]]])
    
    # Add 4 more points with some randomness
    for _ in range(4):
        u = rng.uniform(0.1, 0.9)
        v = rng.uniform(0.1, 0.9 - u)
        bary = np.vstack([bary, [[u, v]]])
    
    return clamp_bary(bary)

def penalty_gradient_descent(bary_init, deadline=None, max_iter=5000):
    """
    Penalty-based gradient descent that directly enforces minimum area constraint.
    Uses explicit sign variables for triangle orientations to handle the absolute value.
    """
    if deadline is not None and time.time() > deadline - 5:
        return bary_init
    
    try:
        bary = bary_init.copy()
        bary = clamp_bary(bary)
        
        # Initialize sign variables based on current orientations
        areas = calc_areas_bary(bary)
        # Determine signs: +1 if det > 0, -1 if det < 0
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
        signs = np.where(det >= 0, 1.0, -1.0)
        
        best_bary = bary.copy()
        best_val = -hard_min_obj(bary.flatten())
        
        lr = 0.01
        penalty_weight = 10.0
        
        t_start = time.time()
        for i in range(max_iter):
            if deadline is not None and time.time() > deadline - 3:
                break
            
            # Compute areas with current signs
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
            signed_areas = signs * det
            min_area = np.min(signed_areas)
            
            # Penalty for negative areas (constraint violation)
            violations = np.maximum(0, -signed_areas)
            penalty = penalty_weight * np.sum(violations ** 2)
            
            # Objective: maximize min_area - penalty
            # Gradient with respect to bary coordinates
            # d(signed_area)/d(bary) = signs * d(det)/d(bary)
            
            # Compute gradient of det with respect to each point's barycentric coords
            grad_u = np.zeros(N)
            grad_v = np.zeros(N)
            
            # For each triangle, add contribution to gradient
            # det = ui*(vj*wk - vk*wj) - vi*(uj*wk - uk*wj) + wi*(uj*vk - uk*vj)
            # where w = 1 - u - v, so dw/du = -1, dw/dv = -1
            
            # Contribution to grad_u[i] from triangle (i,j,k):
            # d(det)/du_i = (vj*wk - vk*wj) + wi*(-1)*(uj*vk - uk*vj) + ui*(-1)*(vj*wk - vk*wj)
            # Simplified: d(det)/du_i = (1-ui)*(vj*wk - vk*wj) - wi*(uj*vk - uk*vj)
            
            # This is complex; let's use numerical gradient for robustness
            eps = 1e-5
            for dim in range(N * 2):
                bary_plus = bary.copy()
                bary_plus[dim // 2, dim % 2] += eps
                bary_plus = clamp_bary(bary_plus)
                
                u_plus = bary_plus[:, 0]
                v_plus = bary_plus[:, 1]
                w_plus = 1.0 - u_plus - v_plus
                
                ui_p = u_plus[tri_idx[:, 0]]
                vi_p = v_plus[tri_idx[:, 0]]
                wi_p = w_plus[tri_idx[:, 0]]
                
                uj_p = u_plus[tri_idx[:, 1]]
                vj_p = v_plus[tri_idx[:, 1]]
                wj_p = w_plus[tri_idx[:, 1]]
                
                uk_p = u_plus[tri_idx[:, 2]]
                vk_p = v_plus[tri_idx[:, 2]]
                wk_p = w_plus[tri_idx[:, 2]]
                
                det_p = ui_p * (vj_p * wk_p - vk_p * wj_p) - vi_p * (uj_p * wk_p - uk_p * wj_p) + wi_p * (uj_p * vk_p - uk_p * vj_p)
                signed_areas_p = signs * det_p
                min_area_p = np.min(signed_areas_p)
                violations_p = np.maximum(0, -signed_areas_p)
                penalty_p = penalty_weight * np.sum(violations_p ** 2)
                obj_p = min_area_p - penalty_p
                
                bary_minus = bary.copy()
                bary_minus[dim // 2, dim % 2] -= eps
                bary_minus = clamp_bary(bary_minus)
                
                u_minus = bary_minus[:, 0]
                v_minus = bary_minus[:, 1]
                w_minus = 1.0 - u_minus - v_minus
                
                ui_m = u_minus[tri_idx[:, 0]]
                vi_m = v_minus[tri_idx[:, 0]]
                wi_m = w_minus[tri_idx[:, 0]]
                
                uj_m = u_minus[tri_idx[:, 1]]
                vj_m = v_minus[tri_idx[:, 1]]
                wj_m = w_minus[tri_idx[:, 1]]
                
                uk_m = u_minus[tri_idx[:, 2]]
                vk_m = v_minus[tri_idx[:, 2]]
                wk_m = w_minus[tri_idx[:, 2]]
                
                det_m = ui_m * (vj_m * wk_m - vk_m * wj_m) - vi_m * (uj_m * wk_m - uk_m * wj_m) + wi_m * (uj_m * vk_m - uk_m * vj_m)
                signed_areas_m = signs * det_m
                min_area_m = np.min(signed_areas_m)
                violations_m = np.maximum(0, -signed_areas_m)
                penalty_m = penalty_weight * np.sum(violations_m ** 2)
                obj_m = min_area_m - penalty_m
                
                grad = (obj_p - obj_m) / (2 * eps)
                bary[dim // 2, dim % 2] += lr * grad
            
            bary = clamp_bary(bary)
            
            # Update signs based on new orientations
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
            new_signs = np.where(det >= 0, 1.0, -1.0)
            signs = new_signs
            
            val = -hard_min_obj(bary.flatten())
            if val > best_val:
                best_val = val
                best_bary = bary.copy()
            
            # Adaptive learning rate
            if i % 100 == 0 and i > 0:
                lr *= 0.95
            if i % 200 == 0 and i > 0:
                penalty_weight *= 0.9
        
        return best_bary
        
    except Exception:
        return bary_init


def try_optimize(bary_init, beta1=50.0, beta2=200.0, maxiter1=200, maxiter2=300, deadline=None):
    """
    Run multi-stage optimization from an initial configuration.
    """
    if deadline is not None and time.time() > deadline - 10:
        return bary_init
    
    try:
        bounds = [(0, 1)] * (N * 2)
        
        # Stage 1: Exploration with smooth landscape
        result1 = minimize(
            soft_min_obj,
            bary_init.flatten(),
            args=(beta1,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter1, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if deadline is not None and time.time() > deadline - 5:
            return result1.x.reshape(N, 2)

        # Stage 2: Exploitation with sharper landscape
        result2 = minimize(
            soft_min_obj,
            result1.x,
            args=(beta2,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter2, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        
        result2_bary = result2.x.reshape(N, 2)
        
        # Stage 3: Penalty-based gradient descent for final polish
        if deadline is not None and time.time() < deadline - 8:
            result3 = penalty_gradient_descent(result2_bary, deadline=deadline, max_iter=3000)
            # Compare and return the better one
            val2 = -hard_min_obj(result2_bary.flatten())
            val3 = -hard_min_obj(result3.flatten())
            if val3 > val2:
                return result3
            else:
                return result2_bary
        
        return result2_bary
        
    except Exception:
        return bary_init

def nelder_mead_polish(bary_init, deadline=None, maxiter=500):
    """
    Polish a configuration using Nelder-Mead on the hard min objective.
    """
    if deadline is not None and time.time() > deadline - 3.0:
        return bary_init
    
    try:
        res_nm = minimize(
            hard_min_obj,
            bary_init.flatten(),
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        return res_nm.x.reshape(N, 2)
    except Exception:
        return bary_init

def simulated_annealing(bary_init, deadline=None, n_iter=2000):
    """
    Simple simulated annealing to escape local minima.
    """
    if deadline is not None and time.time() > deadline - 5.0:
        return bary_init
    
    try:
        bary = bary_init.copy()
        best_bary = bary.copy()
        best_val = -hard_min_obj(bary.flatten())
        
        t_start = time.time()
        t0 = 0.05
        cooling = 0.995
        
        for i in range(n_iter):
            if deadline is not None and time.time() > deadline - 3.0:
                break
            
            # Generate perturbation
            noise = np.random.normal(0, t0, bary.shape)
            new_bary = clamp_bary(bary + noise)
            
            val = -hard_min_obj(bary.flatten())
            new_val = -hard_min_obj(new_bary.flatten())
            
            delta = new_val - val
            if delta > 0 or np.random.random() < np.exp(delta / max(t0, 1e-10)):
                bary = new_bary
                if new_val > best_val:
                    best_val = new_val
                    best_bary = bary.copy()
            
            t0 *= cooling
        
        return best_bary
        
    except Exception:
        return bary_init

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
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

    # 1. Start with hexagonal lattice initialization
    hex_init = generate_hexagonal_init()
    result = try_optimize(hex_init, beta1=50.0, beta2=200.0, maxiter1=300, maxiter2=400, deadline=deadline)
    update_best(result)
    
    # 2. Try structured initialization
    if time.time() < deadline - 30:
        struct_init = generate_structured_init(seed=0)
        result = try_optimize(struct_init, beta1=50.0, beta2=200.0, maxiter1=200, maxiter2=300, deadline=deadline)
        update_best(result)
    
    # 3. Try multiple random initializations
    num_random_starts = 8
    for i in range(num_random_starts):
        if time.time() > deadline - 40:
            break
        rng_seed = 100 + i
        rand_init = generate_random_init(seed=rng_seed)
        result = try_optimize(rand_init, beta1=50.0, beta2=150.0, maxiter1=150, maxiter2=200, deadline=deadline)
        update_best(result)
        
    # 4. Perturb the best found and re-optimize
    if best_bary is not None:
        for i in range(5):
            if time.time() > deadline - 25:
                break
            perturbed = generate_perturbed_init(best_bary, noise_scale=0.02, seed=200 + i)
            result = try_optimize(perturbed, beta1=100.0, beta2=300.0, maxiter1=200, maxiter2=300, deadline=deadline)
            update_best(result)
    
    # 5. Simulated annealing to escape local minima
    if best_bary is not None and time.time() < deadline - 20:
        sa_result = simulated_annealing(best_bary, deadline=deadline, n_iter=1500)
        update_best(sa_result)
    
    # 6. Penalty-based gradient descent for final refinement
    if best_bary is not None and time.time() < deadline - 10:
        pgd_result = penalty_gradient_descent(best_bary, deadline=deadline, max_iter=5000)
        update_best(pgd_result)
    
    # 7. Final polish with Nelder-Mead on hard min
    if best_bary is not None and time.time() < deadline - 3.0:
        nm_result = nelder_mead_polish(best_bary, deadline=deadline, maxiter=500)
        update_best(nm_result)
    
    # Fallback if something went wrong
    if best_bary is None:
        best_bary = generate_random_init(seed=0)
    
    # Final safety clamp and convert to Cartesian
    best_bary = clamp_bary(best_bary)
    result_cart = bary_to_cart(best_bary)
    
    # Ensure all points are strictly inside
    result_cart = np.clip(result_cart, 1e-9, None)
    result_cart[:, 1] = np.clip(result_cart[:, 1], 1e-9, H - 1e-9)
    x_min = result_cart[:, 1] / np.sqrt(3)
    x_max = 1.0 - result_cart[:, 1] / np.sqrt(3)
    result_cart[:, 0] = np.clip(result_cart[:, 0], x_min + 1e-9, x_max - 1e-9)
    
    return result_cart
