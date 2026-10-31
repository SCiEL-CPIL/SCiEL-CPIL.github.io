import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
from itertools import combinations
from scipy.spatial import Delaunay

N = 11
H = np.sqrt(3) / 2

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

def calc_areas_bary(bary):
    """
    Calculate areas of all triangles formed by N points given in barycentric coords (u, v).
    Returns array of shape (165,) containing the area ratios.
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # Extract coordinates for all triangles
    ui = u[tri_idx[:, 0]]
    vi = v[tri_idx[:, 0]]
    wi = w[tri_idx[:, 0]]
    
    uj = u[tri_idx[:, 1]]
    vj = v[tri_idx[:, 1]]
    wj = w[tri_idx[:, 1]]
    
    uk = u[tri_idx[:, 2]]
    vk = v[tri_idx[:, 2]]
    wk = w[tri_idx[:, 2]]
    
    # Determinant calculation vectorized
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    return np.abs(det)

def clamp_bary(bary, eps=1e-7):
    """
    Clamp barycentric coordinates to the strictly valid triangle region.
    u >= eps, v >= eps, u+v <= 1 - eps
    """
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    # Clip to strict interior
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    # Ensure u + v <= 1 - eps
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    # Re-check bounds after scaling
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    return np.column_stack([u, v])

def soft_min_obj(bary_flat, beta=100.0):
    """
    Objective function for optimization.
    Minimizes the negative soft-min of the areas.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    
    areas = calc_areas_bary(bary)
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # To avoid overflow, subtract max
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        return -np.min(areas)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    
    return -soft_min

def hard_min_obj(bary_flat):
    """
    Objective function using hard min.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def bary_to_cart(bary):
    """
    Convert barycentric coords (u, v) to Cartesian (x, y).
    bary: (N, 2) array of (u, v). w = 1 - u - v.
    Point = u*A + v*B + w*C
    A=(0,0), B=(1,0), C=(0.5, H)
    x = v + 0.5*w
    y = w*H
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def cart_to_bary(cart):
    """
    Convert Cartesian coords to barycentric (u, v).
    """
    w = cart[:, 1] / H
    v = cart[:, 0] - 0.5 * w
    u = 1.0 - v - w
    return np.column_stack([u, v])

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration.
    """
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

def generate_perturbed_init(base, noise_scale=0.05, seed=None):
    """Generate perturbed version of a base configuration."""
    rng = np.random.default_rng(seed)
    bary = base.copy()
    bary += rng.normal(0, noise_scale, bary.shape)
    return clamp_bary(bary)

def _local_polish_bary(points_cart, deadline):
    """
    Local refinement in barycentric coordinates using Nelder-Mead.
    """
    bary = cart_to_bary(points_cart)
    bary = clamp_bary(bary)
    
    current_min_area = -hard_min_obj(bary.flatten())
    
    # Nelder-Mead polish with tight tolerances
    if time.time() < deadline - 2:
        try:
            nm_result = minimize(
                hard_min_obj,
                bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True, 'initial_simplex': None}
            )
            new_bary = clamp_bary(nm_result.x.reshape(N, 2))
            new_min_area = -hard_min_obj(new_bary.flatten())
            if new_min_area > current_min_area:
                bary = new_bary
                current_min_area = new_min_area
        except Exception:
            pass
    
    # Convert back to Cartesian
    cart = bary_to_cart(bary)
    return cart, current_min_area

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        """Update best if improved."""
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = -hard_min_obj(bary.flatten())
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # Define bounds for Differential Evolution in barycentric space
    # u, v in [0, 1], with constraint u+v <= 1 handled by clamping in objective
    # But DE needs strict bounds for the sampling.
    # We use [0, 1] for both u and v. The objective function clamps invalid points.
    # To be more efficient, we can restrict v to [0, 1-u] but that's hard for DE.
    # So we stick to [0, 1] x [0, 1] and rely on the objective to penalize/clamp.
    
    bounds = [(0.0, 1.0)] * (N * 2)
    
    # Objective function for DE
    # We use the hard min objective. DE can handle non-smooth functions.
    # To help DE, we can add a small penalty for points on the boundary or outside,
    # but clamp_bary handles the geometry.
    
    def de_objective(x_flat):
        bary = x_flat.reshape(N, 2)
        # Check if points are roughly inside the triangle before calculating areas
        # This is a fast check to avoid wasting time on wildly invalid points
        u = bary[:, 0]
        v = bary[:, 1]
        if np.any(u < -0.1) or np.any(v < -0.1) or np.any(u + v > 1.1):
            return 1.0 # Large penalty
        
        bary_clamped = clamp_bary(bary)
        areas = calc_areas_bary(bary_clamped)
        return -np.min(areas)
    
    # Run Differential Evolution
    # maxiter is the number of generations
    # popsize is the size of the population
    # tol is the tolerance for stopping
    # We give it about 60 seconds to run.
    
    if time.time() < deadline - 30:
        try:
            de_result = differential_evolution(
                de_objective,
                bounds,
                maxiter=100, # Generations
                popsize=15, # Population size
                tol=1e-8,
                mutation=(0.5, 1.5),
                recombination=0.7,
                seed=42,
                polish=False, # We will polish manually
                workers=1, # Single thread to avoid overhead
                update_interval=10 # Report progress every 10 generations (not used here but good practice)
            )
            
            if de_result.success:
                best_bary_de = clamp_bary(de_result.x.reshape(N, 2))
                update_best(best_bary_de)
        except Exception as e:
            print(f"DE failed: {e}")
    
    # If DE didn't find anything or we still have time, try local polish from multiple starts
    # We use the best found so far and some random/hex inits
    
    inits = []
    if best_bary is not None:
        inits.append(best_bary)
    
    # Add hex init
    hex_init = generate_hexagonal_init()
    inits.append(hex_init)
    
    # Add some random inits
    rng = np.random.default_rng(42)
    for i in range(5):
        inits.append(generate_random_init(seed=42 + i))
    
    # Polish each init
    for i, init in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        # Polish using Nelder-Mead
        cart_init = bary_to_cart(init)
        polished_cart, polished_val = _local_polish_bary(cart_init, deadline)
        polished_bary = cart_to_bary(polished_cart)
        update_best(polished_bary)
        
        # If we improved, we might try another round of polish from the new best
        if polished_val > best_val - 1e-10:
             # Re-add the improved point to the queue? No, just continue.
             pass

    # Final global check: if we have time, run a quick DE from the best found
    if best_bary is not None and time.time() < deadline - 15:
        try:
            # Run DE again, but starting near the best solution?
            # DE doesn't support "starting near" easily without custom logic.
            # Instead, we can run a smaller DE with tighter bounds around the best solution.
            # Or just trust the first DE.
            
            # Let's try a second DE run with different parameters to ensure robustness
            de_result2 = differential_evolution(
                de_objective,
                bounds,
                maxiter=50,
                popsize=20,
                tol=1e-9,
                mutation=(0.3, 1.2),
                recombination=0.8,
                seed=123,
                polish=False
            )
            if de_result2.success:
                best_bary_de2 = clamp_bary(de_result2.x.reshape(N, 2))
                update_best(best_bary_de2)
        except Exception:
            pass

    # Convert to Cartesian
    if best_bary is not None:
        cart_points = bary_to_cart(best_bary)
        return cart_points
    else:
        # Fallback
        return bary_to_cart(generate_random_init(0))

# EVOLVE-BLOCK-END
