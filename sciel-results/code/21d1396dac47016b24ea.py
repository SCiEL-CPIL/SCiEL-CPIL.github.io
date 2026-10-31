"""
Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle.
Combined approach: Barycentric coordinates + Soft-Min Objective + L-BFGS-B + Nelder-Mead Polish.
"""

import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

N = 11
H = np.sqrt(3) / 2

# Triangle vertices in Cartesian
# A=(0,0), B=(1,0), C=(0.5, H)
# Barycentric (u, v) where w = 1-u-v
# Point = u*A + v*B + w*C => x = v + 0.5*w, y = w*H

# Pre-compute triangle indices for all combinations of 3 points
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)
_I = tri_idx[:, 0]
_J = tri_idx[:, 1]
_K = tri_idx[:, 2]

def calc_areas_bary(bary):
    """
    Calculate areas of all triangles formed by N points given in barycentric coords (u, v).
    Returns array of shape (165,) containing the area ratios (relative to total triangle area).
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    ui = u[_I]
    vi = v[_I]
    wi = w[_I]
    
    uj = u[_J]
    vj = v[_J]
    wj = w[_J]
    
    uk = u[_K]
    vk = v[_K]
    wk = w[_K]
    
    # Determinant of the matrix [[ui, vi, wi], [uj, vj, wj], [uk, vk, wk]]
    # Area ratio = |det| / 6 (since total area is 1/2 and det relates to 2*Area)
    # Actually, for barycentric coords summing to 1, the area of sub-triangle is |det| * Area_Total.
    # If Area_Total = 1 (normalized), then Area = |det|.
    # Let's verify:
    # If points are vertices (1,0,0), (0,1,0), (0,0,1), det = 1. Area = 1.
    # Wait, the standard barycentric area formula is:
    # Area(PQR) / Area(ABC) = |det( [u_P, v_P, w_P; u_Q, v_Q, w_Q; u_R, v_R, w_R] )|
    # So we just need the absolute determinant.
    
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    return np.abs(det)

def clamp_bary(bary, eps=1e-9):
    """
    Clamp barycentric coordinates to the strictly valid triangle region.
    u >= eps, v >= eps, w >= eps (i.e., u+v <= 1-eps)
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
        # Scale down to fit
        scale = (1 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    # Re-check bounds after scaling to be safe
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    return np.column_stack([u, v])

def soft_min_obj(bary_flat, beta=100.0):
    """
    Objective function for optimization.
    Minimizes the negative soft-min of the areas.
    soft_min(x) = -1/beta * log(sum(exp(-beta * x_i)))
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    
    areas = calc_areas_bary(bary)
    
    # Soft min calculation
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    # Avoid overflow in exp
    if max_scaled > 700:
        return -np.min(areas)
    
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min_val = -log_sum / beta
    
    return -soft_min_val

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

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration in barycentric coords.
    We want points spread out.
    """
    best_bary = None
    best_val = -np.inf
    
    # Try different spacings for a triangular grid
    for s in np.linspace(0.1, 0.4, 15):
        points = []
        # Iterate over a grid in barycentric space
        # u from 0 to 1, v from 0 to 1-u
        # Step size s
        num_steps = int(1.0 / s)
        for i in range(num_steps + 1):
            for j in range(num_steps + 1):
                u = i * s
                v = j * s
                if u + v <= 1.0 and len(points) < N:
                    points.append([u, v])
        
        if len(points) == N:
            bary = np.array(points)
            bary = clamp_bary(bary)
            val = -hard_min_obj(bary.flatten())
            if val > best_val:
                best_val = val
                best_bary = bary.copy()
    
    if best_bary is None:
        # Fallback to random
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

def construct():
    """
    Place 11 points on or inside a unit-area equilateral triangle to maximize
    the smallest area of a triangle formed by three points.
    Returns an array of shape (11, 2) in Cartesian coordinates.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
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

    def try_optimize(bary_init, method='L-BFGS-B', beta1=50.0, beta2=200.0, maxiter1=200, maxiter2=300):
        """
        Two-stage optimization:
        1. Smooth landscape (low beta) to find a good basin.
        2. Sharp landscape (high beta) to refine the minimum.
        """
        if time.time() > deadline - 10:
            return
        
        try:
            # Stage 1: Exploration with Soft Min
            def obj1(x):
                return soft_min_obj(x, beta=beta1)
            
            bounds = [(0, 1)] * (N * 2)
            
            result1 = minimize(
                obj1,
                bary_init.flatten(),
                method=method,
                bounds=bounds,
                options={'maxiter': maxiter1, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            # Check time before stage 2
            if time.time() > deadline - 5:
                update_best(result1.x.reshape(N, 2))
                return

            # Stage 2: Exploitation with sharper Soft Min
            def obj2(x):
                return soft_min_obj(x, beta=beta2)
            
            result2 = minimize(
                obj2,
                result1.x,
                method=method,
                bounds=bounds,
                options={'maxiter': maxiter2, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            update_best(result2.x.reshape(N, 2))
                
        except Exception:
            pass

    # 1. Start with hexagonal lattice initialization
    hex_init = generate_hexagonal_init()
    try_optimize(hex_init, method='L-BFGS-B', beta1=50.0, beta2=200.0, maxiter1=300, maxiter2=400)
    
    # 2. Try multiple random initializations
    num_random_starts = 10
    for i in range(num_random_starts):
        if time.time() > deadline - 30:
            break
        rng_seed = 100 + i
        rand_init = generate_random_init(seed=rng_seed)
        # Use slightly fewer iterations for random starts to allow more of them
        try_optimize(rand_init, method='L-BFGS-B', beta1=50.0, beta2=150.0, maxiter1=150, maxiter2=200)
        
    # 3. Perturb the best found and re-optimize
    if best_bary is not None:
        for i in range(8):
            if time.time() > deadline - 20:
                break
            perturbed = generate_perturbed_init(best_bary, noise_scale=0.02, seed=200 + i)
            try_optimize(perturbed, method='L-BFGS-B', beta1=100.0, beta2=300.0, maxiter1=200, maxiter2=300)
    
    # 4. Final polish with Nelder-Mead on hard min
    # Nelder-Mead doesn't use gradients, so it can handle the non-differentiable min better at the end
    if best_bary is not None and time.time() < deadline - 10:
        try:
            nm_result = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            update_best(nm_result.x.reshape(N, 2))
        except Exception:
            pass
            
    # 5. Convert best barycentric to Cartesian
    if best_bary is not None:
        result_cart = bary_to_cart(best_bary)
        # Final clamp to ensure strictly inside for safety, though bary clamp should handle it
        # The evaluator expects points on or inside.
        return result_cart
    else:
        # Fallback: return a valid configuration
        fallback_bary = generate_random_init(seed=0)
        return bary_to_cart(fallback_bary)

# EVOLVE-BLOCK-END
