"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

N = 11

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)
NUM_TRIS = len(tri_indices)

# Triangle vertices in Cartesian
H = np.sqrt(3) / 2

def bary_to_cart(bary):
    """
    Convert barycentric coords (u, v) to Cartesian (x, y).
    Point = u*A + v*B + w*C where A=(0,0), B=(1,0), C=(0.5, H), w = 1-u-v
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def cart_to_bary(cart):
    """Convert Cartesian to barycentric (u, v)."""
    x = cart[:, 0]
    y = cart[:, 1]
    v = x - y / (2 * H)
    u = 1 - x - y / (2 * H)
    return np.column_stack([u, v])

def calc_areas_bary(bary):
    """
    Calculate areas of all triangles formed by N points in barycentric coords.
    Returns array of shape (165,) containing the area ratios.
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

def clamp_bary(bary, eps=1e-6):
    """Clamp barycentric coordinates to the valid triangle region."""
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
    """Objective: negative of minimum triangle area (to minimize)."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def soft_min_obj(bary_flat, beta=100.0):
    """Smooth approximation of the min using log-sum-exp."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return log_sum / beta

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
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

    # Phase 1: Differential Evolution Global Search
    # We search in the space of barycentric coordinates (u, v) for each point.
    # Bounds are [0, 1] for each coordinate.
    # The objective function clamps invalid points to the triangle.
    
    try:
        bounds = [(0, 1)] * (N * 2)
        
        de_result = differential_evolution(
            hard_min_obj,
            bounds=bounds,
            maxiter=200,
            popsize=30,
            tol=1e-8,
            seed=42,
            polish=False,
            init='latinhypercube'
        )
        
        if -de_result.fun > best_val:
            update_best(de_result.x.reshape(N, 2))
    except Exception:
        pass

    # Phase 2: Local Polish with L-BFGS-B on Soft Min
    if best_bary is not None:
        try:
            if time.time() < deadline - 10:
                result = minimize(
                    soft_min_obj,
                    best_bary.flatten(),
                    args=(200.0,),
                    method='L-BFGS-B',
                    bounds=[(0, 1)] * (N * 2),
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                update_best(result.x.reshape(N, 2))
        except Exception:
            pass

    # Phase 3: Try a few random restarts with L-BFGS if time permits
    for i in range(5):
        if time.time() > deadline - 20:
            break
        try:
            rng = np.random.default_rng(100 + i)
            u = rng.random(N)
            v = rng.random(N) * (1 - u)
            init_bary = np.column_stack([u, v])
            
            result = minimize(
                soft_min_obj,
                init_bary.flatten(),
                args=(100.0,),
                method='L-BFGS-B',
                bounds=[(0, 1)] * (N * 2),
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            update_best(result.x.reshape(N, 2))
        except Exception:
            pass

    # Final Polish with Nelder-Mead on Hard Min
    if best_bary is not None and time.time() < deadline - 5:
        try:
            result = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            update_best(result.x.reshape(N, 2))
        except Exception:
            pass

    if best_bary is None:
        rng = np.random.default_rng(0)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])

    bary_final = clamp_bary(best_bary, eps=1e-5)
    cart = bary_to_cart(bary_final)
    
    return cart
