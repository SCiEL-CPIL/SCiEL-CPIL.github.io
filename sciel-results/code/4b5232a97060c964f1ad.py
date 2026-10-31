from scipy.optimize import differential_evolution
import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 148.0

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
    """Convert Cartesian (x, y) to barycentric (u, v)."""
    x = cart[:, 0]
    y = cart[:, 1]
    v = 2.0 * y / H
    u = 1.0 - v - x
    return np.column_stack([u, v])

def calc_min_area_cartesian(points):
    """
    Calculate the minimum area of all triangles formed by N points.
    points: (N, 2) array of Cartesian coordinates.
    """
    p_i = points[tri_idx[:, 0]]
    p_j = points[tri_idx[:, 1]]
    p_k = points[tri_idx[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_k[:, 0] - p_i[:, 0]) * (p_j[:, 1] - p_i[:, 1])
    
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

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

def hard_min_obj_bary(bary_flat):
    """Objective using hard min in barycentric coords."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    cart = bary_to_cart(bary)
    return -calc_min_area_cartesian(cart)

def generate_lattice_init():
    """Generate a triangular lattice initialization with 10 points + 1 center."""
    pts = []
    for i in range(5):
        for j in range(5 - i):
            pts.append([i / 4.0, j / 4.0])
    pts = np.array(pts)
    
    center = np.array([1/3, 1/3])
    pts = np.vstack([pts, center])
    
    return pts

def generate_hexagonal_init():
    """Generate a hexagonal packing-inspired initialization."""
    s = 0.25
    hex_pts = []
    for k in range(6):
        angle = k * np.pi / 3
        hex_pts.append([s * np.cos(angle), s * np.sin(angle)])
    
    inner_pts = []
    inner_pts.append([0.0, 0.0])
    for k in range(4):
        angle = k * np.pi / 2 + np.pi / 4
        inner_pts.append([0.15 * np.cos(angle), 0.15 * np.sin(angle)])
    
    all_pts = np.array(hex_pts + inner_pts)
    bary = cart_to_bary(all_pts)
    bary = clamp_bary(bary)
    return bary

def generate_random_init(seed=None):
    """Generate random valid barycentric coordinates."""
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def generate_perturbed_init(base, scale=0.05, seed=None):
    """Generate perturbed version of base configuration."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, scale, base.shape)
    perturbed = base + noise
    return clamp_bary(perturbed)

def refine_local(bary_init, time_limit=5.0, use_soft_min=False):
    """
    Refine a configuration using local optimization.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Nelder-Mead on hard min (robust to non-smoothness)
    try:
        obj_func = hard_min_obj_bary
        result1 = minimize(
            obj_func,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        bary_current = clamp_bary(result1.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.5:
        return bary_current
    
    # Stage 2: SLSQP polish with hard min for precise convergence
    try:
        result2 = minimize(
            hard_min_obj_bary,
            bary_current.flatten(),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14}
        )
        bary_current = clamp_bary(result2.x.reshape(N, 2))
    except Exception:
        pass
        
    return bary_current

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    good_candidates = []
    
    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        cart = bary_to_cart(bary)
        val = calc_min_area_cartesian(cart)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    def track_good(bary):
        nonlocal best_val
        bary = clamp_bary(bary)
        cart = bary_to_cart(bary)
        val = calc_min_area_cartesian(cart)
        if len(good_candidates) < 5 or val > best_val * 0.98:
            good_candidates.append((val, bary.copy()))
            good_candidates.sort(key=lambda x: -x[0])
            del good_candidates[5:]

    # 1. Lattice Init
    lat_init = generate_lattice_init()
    refined_lat = refine_local(lat_init, time_limit=8.0)
    update_best(refined_lat)
    track_good(refined_lat)
    
    # 1b. Hexagonal Init
    if time.time() < deadline - 60:
        hex_init = generate_hexagonal_init()
        refined_hex = refine_local(hex_init, time_limit=6.0)
        update_best(refined_hex)
        track_good(refined_hex)
    
    # 2. Global Search with Differential Evolution
    if time.time() < deadline - 40:
        try:
            def de_obj(x):
                bary = x.reshape(N, 2)
                bary = clamp_bary(bary)
                cart = bary_to_cart(bary)
                return -calc_min_area_cartesian(cart)
            
            bounds = [(0, 1)] * (N * 2)
            result_de = differential_evolution(
                de_obj,
                bounds,
                seed=42,
                maxiter=35,
                popsize=15,
                tol=1e-4,
                polish=False
            )
            
            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined_de = refine_local(de_bary, time_limit=8.0)
            update_best(refined_de)
            track_good(refined_de)
        except Exception:
            pass
    
    # 3. Basin Hopping (manual loop for time safety)
    if time.time() < deadline - 10 and best_bary is not None:
        current_bary = best_bary.copy()
        current_val = best_val
        rng_bh = np.random.default_rng(42)
        
        while time.time() < deadline - 5:
            noise = rng_bh.normal(0, 0.2, current_bary.shape)
            new_bary = clamp_bary(current_bary + noise)
            
            refined = refine_local(new_bary, time_limit=min(3.0, deadline - 5 - time.time()))
            val = calc_min_area_cartesian(bary_to_cart(refined))
            
            if val > current_val:
                current_val = val
                current_bary = refined
                update_best(refined)
                track_good(refined)
    
    cart = bary_to_cart(best_bary)
    return cart
