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

def energy_obj(bary_flat, eps=1e-6):
    """
    Objective: Sum of inverse squared areas.
    This creates a smooth landscape where small triangles are heavily penalized.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    # Avoid division by zero
    areas_safe = areas + eps
    return np.sum(1.0 / (areas_safe * areas_safe))

def hard_min_obj(bary_flat):
    """Objective using hard min."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

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

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using two stages:
    1. L-BFGS-B on the smooth energy function (sum of inverse squares).
    2. Nelder-Mead on the hard minimum.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    # Stage 1: Smooth energy minimization
    try:
        result1 = minimize(
            energy_obj,
            bary_init.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
    except Exception:
        return bary_init
    
    if time.time() - start_time > time_limit:
        return bary_init
        
    bary_mid = clamp_bary(result1.x.reshape(N, 2))
    
    # Stage 2: Hard min refinement with Nelder-Mead
    try:
        result2 = minimize(
            hard_min_obj,
            bary_mid.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-14}
        )
    except Exception:
        return bary_mid
        
    return clamp_bary(result2.x.reshape(N, 2))

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
    refined_hex = refine_local(hex_init, time_limit=10.0)
    update_best(refined_hex)
    
    # 2. Global Search with Differential Evolution
    if time.time() < deadline - 30:
        try:
            def de_obj(x):
                bary = x.reshape(N, 2)
                bary = clamp_bary(bary)
                areas = calc_areas_bary(bary)
                return -np.min(areas)
            
            bounds = [(0, 1)] * (N * 2)
            # Use a smaller population and fewer iterations to fit in time
            result_de = differential_evolution(
                de_obj,
                bounds,
                seed=42,
                maxiter=50,
                popsize=10,
                tol=1e-8,
                polish=False # We will polish manually
            )
            
            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined_de = refine_local(de_bary, time_limit=10.0)
            update_best(refined_de)
        except Exception:
            pass
    
    # 3. Random Restarts with Local Refinement
    rng = np.random.default_rng(123)
    restart_count = 0
    while time.time() < deadline - 10:
        restart_count += 1
        rand_init = generate_random_init(seed=1000 + restart_count)
        # Use a shorter time limit for random restarts to allow more of them
        refined_rand = refine_local(rand_init, time_limit=5.0)
        update_best(refined_rand)
        
        # Occasionally try a perturbation of the best solution
        if best_bary is not None and restart_count % 5 == 0:
            noise = rng.normal(0, 0.05, best_bary.shape)
            perturbed = clamp_bary(best_bary + noise)
            refined_pert = refine_local(perturbed, time_limit=5.0)
            update_best(refined_pert)

    # 4. Final Polish on the best solution found
    if best_bary is not None and time.time() < deadline - 2:
        # Try a very aggressive Nelder-Mead on the best
        try:
            result_final = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-16}
            )
            update_best(clamp_bary(result_final.x.reshape(N, 2)))
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
