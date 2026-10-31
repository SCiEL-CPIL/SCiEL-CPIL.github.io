import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
from itertools import combinations

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

def cart_to_bary(cart):
    """Convert Cartesian (x, y) to barycentric (u, v)."""
    x = cart[:, 0]
    y = cart[:, 1]
    S = 1 - y / H
    u = 0.5 + 0.5 * S - x
    v = S - u
    return np.column_stack([u, v])

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

def calc_min_area(bary):
    """Calculate minimum area efficiently."""
    return np.min(calc_areas_bary(bary))

def clamp_bary(bary, eps=1e-10):
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

def generate_initial_points():
    """
    Generate diverse high-quality initial points.
    Uses a combination of lattice, random, and repulsive force methods.
    """
    inits = []
    rng = np.random.default_rng(42)
    
    # 1. Lattice-based initialization
    # Try to place points on a triangular grid
    for scale in [0.1, 0.15, 0.2, 0.25, 0.3]:
        center = np.array([1.0/3, 1.0/3])
        cart_center = bary_to_cart(center.reshape(1,2))[0]
        
        cart_pts = [cart_center.copy()]
        
        # 6 neighbors
        for k in range(6):
            theta = np.pi/3 * k
            dx = scale * np.cos(theta)
            dy = scale * np.sin(theta)
            cart_pts.append(cart_center + np.array([dx, dy]))
            
        # Next ring
        for k in range(6):
            theta = np.pi/3 * k + np.pi/6
            dx = 2 * scale * np.cos(theta)
            dy = 2 * scale * np.sin(theta)
            cart_pts.append(cart_center + np.array([dx, dy]))
            
        if len(cart_pts) >= N:
            cart_pts = np.array(cart_pts[:N])
            bary_pts = cart_to_bary(cart_pts)
            bary_pts = clamp_bary(bary_pts)
            if np.all(bary_pts[:, 0] > 0) and np.all(bary_pts[:, 1] > 0) and np.all(bary_pts[:, 0] + bary_pts[:, 1] < 1):
                inits.append(bary_pts)

    # 2. Repulsive force initialization
    # Start with random points and let them repel
    for _ in range(10):
        cart_pts = rng.random((N, 2))
        # Scale to fit in triangle roughly
        cart_pts = cart_pts * 0.8 + 0.1
        bary_pts = cart_to_bary(cart_pts)
        bary_pts = clamp_bary(bary_pts)
        
        # Simple repulsion step
        for _ in range(100):
            forces = np.zeros_like(bary_pts)
            for i in range(N):
                for j in range(N):
                    if i == j: continue
                    diff = bary_pts[i] - bary_pts[j]
                    dist_sq = np.sum(diff**2)
                    if dist_sq < 1e-6:
                        dist_sq = 1e-6
                    dist = np.sqrt(dist_sq)
                    # Force proportional to 1/dist^2
                    force_mag = 0.1 / dist_sq
                    forces[i] += force_mag * diff / dist
            bary_pts += 0.01 * forces
            bary_pts = clamp_bary(bary_pts)
        
        inits.append(bary_pts)

    # 3. Random
    for _ in range(10):
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        inits.append(np.column_stack([u, v]))
        
    return inits

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using a robust multi-stage optimization.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    curr_val = calc_min_area(bary_current)
    
    # Stage 1: Nelder-Mead on hard min (good for non-smooth)
    if time.time() - start_time < time_limit * 0.3:
        try:
            result_nm = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            new_bary = clamp_bary(result_nm.x.reshape(N, 2))
            new_val = calc_min_area(new_bary)
            if new_val > curr_val:
                bary_current = new_bary
                curr_val = new_val
        except Exception:
            pass

    # Stage 2: L-BFGS-B on soft min (smooths the landscape)
    # Use increasing beta to focus on the minimum
    for beta in [50.0, 200.0, 500.0]:
        if time.time() - start_time > time_limit * 0.7:
            break
        try:
            result_lbfgs = minimize(
                soft_min_obj,
                bary_current.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            new_bary = clamp_bary(result_lbfgs.x.reshape(N, 2))
            new_val = calc_min_area(new_bary)
            if new_val > curr_val:
                bary_current = new_bary
                curr_val = new_val
        except Exception:
            pass

    # Stage 3: SLSQP on hard min (precise)
    if time.time() - start_time < time_limit:
        try:
            result_slsqp = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-16}
            )
            new_bary = clamp_bary(result_slsqp.x.reshape(N, 2))
            new_val = calc_min_area(new_bary)
            if new_val > curr_val:
                bary_current = new_bary
                curr_val = new_val
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
        val = calc_min_area(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # Generate initial points
    inits = generate_initial_points()
    
    # Sort by initial quality
    inits_with_vals = [(init, calc_min_area(init)) for init in inits]
    inits_with_vals.sort(key=lambda x: -x[1])
    inits = [x[0] for x in inits_with_vals]
    
    # Refine each initial point
    for i, init in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        # Allocate time
        remaining_inits = len(inits) - i
        time_per_init = min(5.0, (deadline - time.time() - 10) / max(1, remaining_inits))
        if time_per_init < 0.5:
            time_per_init = 0.5
            
        refined = refine_local(init, time_limit=time_per_init)
        update_best(refined)
        
    # Final polish on the best found
    if best_bary is not None and time.time() < deadline - 5:
        try:
            bounds = [(0, 1)] * (N * 2)
            # One more Nelder-Mead pass
            result_nm = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            if result_nm.fun < hard_min_obj(best_bary.flatten()):
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
        except Exception:
            pass
        
        if best_bary is not None and time.time() < deadline - 2:
            try:
                # SLSQP polish
                result_slsqp = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 5000, 'ftol': 1e-18}
                )
                if result_slsqp.fun < hard_min_obj(best_bary.flatten()):
                    update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
