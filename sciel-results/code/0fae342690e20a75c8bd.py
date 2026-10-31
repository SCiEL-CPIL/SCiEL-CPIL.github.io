import numpy as np
from scipy.optimize import minimize, differential_evolution
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

def epigraph_obj(params, beta=100.0):
    """
    Epigraph formulation: maximize t such that all areas >= t.
    params = [u1, v1, ..., uN, vN, t]
    Uses soft constraints: penalize when area < t.
    """
    bary_flat = params[:2*N]
    t = params[2*N]
    
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Penalty for areas below t
    deficit = np.maximum(0, t - areas)
    penalty = beta * np.sum(deficit**2)
    
    # We want to maximize t, so minimize -t + penalty
    return -t + penalty

def generate_hexagonal_init():
    """
    Generate a hexagonal grid pattern inside the triangle.
    This provides a well-spaced starting point.
    """
    # Create a hexagonal lattice in barycentric coordinates
    # Use a 3x3 grid with some jitter
    rng = np.random.default_rng(123)
    
    # Base hexagonal positions in barycentric coords
    # Center of triangle is (1/3, 1/3)
    points = []
    
    # Add center
    points.append([1/3, 1/3])
    
    # Add 6 points around center in hexagonal pattern
    r = 0.2
    for angle in np.linspace(0, 2*np.pi, 7)[:-1]:
        u = 1/3 + r * np.cos(angle)
        v = 1/3 + r * np.sin(angle) * 0.866  # Adjust for triangular geometry
        points.append([u, v])
    
    # Add 4 more points to reach 11
    # Place them near the edges
    edge_points = [
        [0.1, 0.1],      # Near bottom-left
        [0.5, 0.1],      # Near bottom-center  
        [0.9, 0.1],      # Near bottom-right
        [0.15, 0.5],     # Left side
    ]
    points.extend(edge_points)
    
    # Convert to numpy and clamp
    bary = np.array(points)
    bary = clamp_bary(bary)
    
    # Add small random perturbation to break symmetry
    bary += rng.random(bary.shape) * 0.02
    bary = clamp_bary(bary)
    
    return bary

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages with epigraph formulation.
    """
    start_time = time.time()
    
    bary_current = bary_init.copy()
    
    # Initialize t with current minimum area
    areas = calc_areas_bary(bary_current)
    t_init = np.min(areas) * 0.95
    
    # Build initial params: [bary_coords..., t]
    params = np.concatenate([bary_current.flatten(), [t_init]])
    
    # Bounds: bary coords in [0,1], t in [0, 0.1]
    bounds = [(0, 1)] * (2*N) + [(0, 0.1)]
    
    # Stage 1: Epigraph with moderate beta
    try:
        result1 = minimize(
            epigraph_obj,
            params,
            args=(50.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result1.success or result1.fun < epigraph_obj(params, 50.0):
            bary_current = clamp_bary(result1.x[:2*N].reshape(N, 2))
            t_init = result1.x[2*N]
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.2:
        return bary_current

    # Stage 2: Epigraph with higher beta (sharper constraints)
    params = np.concatenate([bary_current.flatten(), [t_init]])
    try:
        result2 = minimize(
            epigraph_obj,
            params,
            args=(200.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-9}
        )
        if result2.success or result2.fun < epigraph_obj(params, 200.0):
            bary_current = clamp_bary(result2.x[:2*N].reshape(N, 2))
            t_init = result2.x[2*N]
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.4:
        return bary_current

    # Stage 3: Very sharp epigraph
    params = np.concatenate([bary_current.flatten(), [t_init]])
    try:
        result3 = minimize(
            epigraph_obj,
            params,
            args=(1000.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if result3.success or result3.fun < epigraph_obj(params, 1000.0):
            bary_current = clamp_bary(result3.x[:2*N].reshape(N, 2))
            t_init = result3.x[2*N]
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.6:
        return bary_current

    # Stage 4: Hard min refinement with Nelder-Mead
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
        
    if time.time() - start_time > time_limit * 0.8:
        return bary_current

    # Stage 5: Final high-precision Nelder-Mead polish
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

    # 1. Try hexagonal initialization first (structured, well-spaced)
    hex_bary = generate_hexagonal_init()
    update_best(hex_bary)
    
    # 2. Global Search using Differential Evolution with soft-min
    bounds = [(0, 1)] * (N * 2)
    
    # DE callback to check time
    def de_callback(xk, convergence):
        return time.time() > deadline - 65  # Stop DE if we need time for local refinement
    
    # Use DE with moderate beta
    try:
        de_result = differential_evolution(
            soft_min_obj,
            bounds,
            args=(50.0,),  # Moderate beta for DE
            maxiter=80,
            popsize=15,
            tol=1e-8,
            seed=42,
            polish=False,  # We do our own polishing
            init='sobol', # Better initial distribution
            callback=de_callback
        )
        
        if de_result.success or de_result.fun < -np.inf:
            de_bary = clamp_bary(de_result.x.reshape(N, 2))
            update_best(de_bary)
            
    except Exception as e:
        pass

    # 3. Local Refinement of the best result
    if best_bary is not None:
        if time.time() < deadline - 15:
            refined = refine_local(best_bary, time_limit=min(35.0, deadline - 15 - time.time()))
            update_best(refined)

    # 4. Try additional random starts if time permits
    if time.time() < deadline - 25:
        for seed in [1, 2, 3, 4]:
            if time.time() > deadline - 12:
                break
            rng = np.random.default_rng(seed)
            u = rng.random(N)
            v = rng.random(N) * (1 - u)
            rand_bary = np.column_stack([u, v])
            
            refined_rand = refine_local(rand_bary, time_limit=min(15.0, deadline - 12 - time.time()))
            update_best(refined_rand)

    # 5. Final Polish if time allows
    if best_bary is not None and time.time() < deadline - 5:
        final_polish = refine_local(best_bary, time_limit=min(8.0, deadline - 5 - time.time()))
        update_best(final_polish)

    if best_bary is None:
        # Absolute fallback
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])

    cart = bary_to_cart(best_bary)
    return cart

if __name__ == "__main__":
    pts = construct()
    print(pts)
