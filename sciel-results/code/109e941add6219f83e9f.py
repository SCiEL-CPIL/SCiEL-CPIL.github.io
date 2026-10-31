"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

N = 11

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

def calc_areas_and_grad(bary):
    """Calculate areas and gradient of areas w.r.t. barycentric coords."""
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
    areas = np.abs(det)
    
    # Gradient of det w.r.t. u_i, v_i for each triangle
    # d(det)/du_i = vj*wk - vk*wj
    # d(det)/dv_i = uk*wi - ui*wk
    # d(det)/du_j = vi*wk - vk*wi
    # d(det)/dv_j = uk*wi - ui*wk  (wait, let me recompute)
    
    # Actually, let's compute gradient properly
    # det = ui*(vj*wk - vk*wj) - vi*(uj*wk - uk*wj) + wi*(uj*vk - uk*vj)
    
    # d(det)/dui = vj*wk - vk*wj
    # d(det)/dvi = -(uj*wk - uk*wj) = uk*wj - uj*wk
    # d(det)/duj = -vi*wk + wi*vk = wi*vk - vi*wk
    # d(det)/dvj = ui*wk - wi*uk
    # d(det)/duk = -vi*wj + wi*vj = wi*vj - vi*wj
    # d(det)/dvk = ui*wj - wi*uj
    
    grad_det = np.zeros_like(bary)  # (N, 2)
    
    # For each triangle, add contribution to gradient
    for t in range(len(tri_idx)):
        i, j, k = tri_idx[t]
        sign = np.sign(det[t]) if det[t] != 0 else 1.0
        
        # d(det)/dui
        grad_det[i, 0] += sign * (vj[t]*wk[t] - vk[t]*wj[t])
        # d(det)/dvi
        grad_det[i, 1] += sign * (uk[t]*wj[t] - uj[t]*wk[t])
        # d(det)/duj
        grad_det[j, 0] += sign * (wi[t]*vk[t] - vi[t]*wk[t])
        # d(det)/dvj
        grad_det[j, 1] += sign * (ui[t]*wk[t] - wi[t]*uk[t])
        # d(det)/duk
        grad_det[k, 0] += sign * (wi[t]*vj[t] - vi[t]*wj[t])
        # d(det)/dvk
        grad_det[k, 1] += sign * (ui[t]*wj[t] - wi[t]*uj[t])
    
    return areas, grad_det

def clamp_bary(bary, eps=1e-6):
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

def soft_min_obj(bary_flat, beta=80.0):
    """
    Smooth surrogate for the hard min via log-sum-exp.
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

def hard_min_obj(bary_flat):
    """Objective using hard min."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def gradient_ascent_step(bary, learning_rate=0.01, num_small=10):
    """
    Custom gradient ascent that prioritizes increasing the smallest triangle areas.
    This acts like a max-flow balancing approach: push points away from configurations
    that create small triangles, with emphasis on the most constrained ones.
    """
    areas, grad_det = calc_areas_and_grad(bary)
    
    # Find the smallest triangles
    sorted_idx = np.argsort(areas)
    small_idx = sorted_idx[:num_small]
    
    # Weighted gradient: emphasize smallest triangles
    weights = np.zeros_like(areas)
    min_area = areas[small_idx[0]]
    for idx in small_idx:
        # Inverse area weighting (with floor to avoid division by zero)
        weights[idx] = 1.0 / (areas[idx] + 1e-10)
    
    # Normalize weights
    weights = weights / (np.sum(weights) + 1e-10)
    
    # Compute weighted gradient
    weighted_grad = np.zeros_like(bary)
    for t in range(len(tri_idx)):
        if weights[t] > 0:
            i, j, k = tri_idx[t]
            weighted_grad[i, 0] += weights[t] * grad_det[i, 0]
            weighted_grad[i, 1] += weights[t] * grad_det[i, 1]
            weighted_grad[j, 0] += weights[t] * grad_det[j, 0]
            weighted_grad[j, 1] += weights[t] * grad_det[j, 1]
            weighted_grad[k, 0] += weights[t] * grad_det[k, 0]
            weighted_grad[k, 1] += weights[t] * grad_det[k, 1]
    
    # Gradient ascent step
    bary_new = bary + learning_rate * weighted_grad
    bary_new = clamp_bary(bary_new)
    
    return bary_new

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple stages:
    1. Gradient ascent with emphasis on smallest triangles
    2. L-BFGS-B on the smooth soft-min surrogate
    3. Nelder-Mead on the hard minimum
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    # Stage 0: Gradient ascent to balance triangle areas
    bary_grad = bary_init.copy()
    for step in range(50):
        if time.time() - start_time > time_limit * 0.3:
            break
        lr = 0.005 * (0.95 ** step)  # Decaying learning rate
        bary_grad = gradient_ascent_step(bary_grad, learning_rate=lr, num_small=15)
    
    # Stage 1: Smooth soft-min minimization
    try:
        result1 = minimize(
            soft_min_obj,
            bary_grad.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
        )
    except Exception:
        result1 = type('obj', (), {'x': bary_grad.flatten()})()
    
    if time.time() - start_time > time_limit * 0.7:
        return clamp_bary(result1.x.reshape(N, 2))
        
    bary_mid = clamp_bary(result1.x.reshape(N, 2))
    
    # Stage 2: Hard min refinement with Nelder-Mead
    try:
        result2 = minimize(
            hard_min_obj,
            bary_mid.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 800, 'xatol': 1e-12, 'fatol': 1e-16}
        )
    except Exception:
        return bary_mid
        
    return clamp_bary(result2.x.reshape(N, 2))

def generate_hexagonal_init():
    """Generate a hexagonal lattice-like initial configuration."""
    best_bary = None
    best_val = -np.inf
    
    for s in np.linspace(0.1, 0.5, 20):
        points = []
        for i in range(5):
            for j in range(5):
                u = i * s
                v = j * s * np.sqrt(3) / 2
                if u + v <= 1 and len(points) < N:
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

def generate_perturbed_init(base_bary, sigma=0.08, seed=None):
    """Generate a perturbed version of a base configuration."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, base_bary.shape)
    return clamp_bary(base_bary + noise)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
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
    
    # 2. Global Search with Differential Evolution (shorter budget)
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
                maxiter=30,
                popsize=10,
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
    while time.time() < deadline - 15:
        restart_count += 1
        if best_bary is not None and restart_count % 3 == 0:
            # Perturbation of best solution
            init = generate_perturbed_init(best_bary, sigma=0.06, seed=2000 + restart_count)
        else:
            # Fresh random start
            init = generate_random_init(seed=3000 + restart_count)
        
        refined = refine_local(init, time_limit=4.0)
        update_best(refined)
    
    # 4. Final high-precision polish: SLSQP then Nelder-Mead
    if best_bary is not None and time.time() < deadline - 3:
        try:
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
        
        if best_bary is not None and time.time() < deadline - 1:
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-18}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass

    if best_bary is None:
        best_bary = generate_random_init(seed=42)
    
    bary_final = clamp_bary(best_bary, eps=1e-5)
    cart = bary_to_cart(bary_final)
    
    # Safety check to ensure points are strictly inside
    for i in range(N):
        x, y = cart[i]
        if y < 1e-9:
            cart[i, 1] = 1e-9
        if y > np.sqrt(3) * x - 1e-9:
            cart[i, 0] = max(cart[i, 0], y / np.sqrt(3) + 1e-6)
        if y + np.sqrt(3) * x > np.sqrt(3) - 1e-9:
            cart[i, 0] = min(cart[i, 0], (np.sqrt(3) - y) / np.sqrt(3) - 1e-6)
    
    return cart
