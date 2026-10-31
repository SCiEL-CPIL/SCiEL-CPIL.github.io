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
AREA_EQUI = 0.5 * H  # Area of the unit equilateral triangle

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

def hard_min_obj(x):
    """
    Objective function for minimization.
    We want to maximize the minimum area, so we minimize the negative of the minimum area.
    """
    bary = x.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def generate_lattice_init(scale=1.0, offset_u=0.0, offset_v=0.0):
    """
    Generate a configuration based on a triangular lattice scaled to fit the triangle.
    This provides a structured, spread-out starting point.
    """
    pts = []
    
    # Triangular lattice points in barycentric coordinates
    # We want to place 11 points.
    # A common good configuration is derived from a 4x4 grid (10 points) + 1 center,
    # or a specific hexagonal packing.
    
    # Let's try a grid of points (i, j) such that i+j <= 3 (10 points)
    # and add the centroid or a perturbed point.
    
    # Actually, for N=11, the optimal is often close to:
    # 3 points on each side (excluding vertices? no, including vertices is usually bad for max min area)
    # Let's stick to the 10-point grid + 1 random/perturbed point, but scale it.
    
    # Standard 10-point grid:
    for i in range(4):
        for j in range(4 - i):
            pts.append([i/3.0, j/3.0])
    
    # Add 11th point. The "hole" in the middle is around (0.5, 0.25) in barycentric?
    # Centroid is (1/3, 1/3).
    # Let's place the 11th point at the centroid of the central empty region.
    # The central region is a hexagon or triangle.
    # Let's just place it at (0.5, 0.25) which is roughly center.
    pts.append([0.5, 0.25])
    
    bary = np.array(pts)
    
    # Apply scale and offset to explore variations
    # Scale towards center (1/3, 1/3)
    center = np.array([1.0/3.0, 1.0/3.0])
    bary = center + scale * (bary - center)
    
    bary[:, 0] += offset_u
    bary[:, 1] += offset_v
    
    return clamp_bary(bary)

def refine_local(bary, time_limit=5.0):
    """
    Perform a local refinement using Nelder-Mead to polish the solution.
    """
    start_time = time.time()
    
    try:
        res = minimize(
            hard_min_obj,
            bary.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        
        # Check time
        if time.time() > start_time + time_limit:
            return bary
            
        candidate_bary = clamp_bary(res.x.reshape(N, 2))
        return candidate_bary
    except Exception:
        return bary

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_val = -np.inf
    best_bary = None
    
    # Strategy:
    # 1. Generate a set of diverse initial configurations based on lattice structures.
    # 2. Refine each one locally.
    # 3. Keep the best.
    
    # List of initial configurations to try
    inits = []
    
    # 1. Standard 10-point grid + center
    inits.append(generate_lattice_init(1.0, 0.0, 0.0))
    
    # 2. Scaled versions (pulling points inwards)
    for scale in [0.9, 0.8, 0.7]:
        inits.append(generate_lattice_init(scale, 0.0, 0.0))
        
    # 3. Perturbed versions (shifting the whole lattice)
    for du in [-0.05, 0.0, 0.05]:
        for dv in [-0.05, 0.0, 0.05]:
            if du == 0 and dv == 0: continue
            inits.append(generate_lattice_init(1.0, du, dv))
            
    # 4. Random restarts (but fewer, just to be safe)
    rng = np.random.default_rng(42)
    for _ in range(5):
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        inits.append(clamp_bary(np.column_stack([u, v])))
        
    # Sort inits by their initial objective value (best first) to prioritize good starts
    init_scores = []
    for bary in inits:
        val = -hard_min_obj(bary.flatten())
        init_scores.append((val, bary))
        
    init_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Refine the top configurations
    for val, bary in init_scores:
        if time.time() > deadline - 10.0:
            break
            
        # Refine
        polished_bary = refine_local(bary, time_limit=10.0)
        polished_val = -hard_min_obj(polished_bary.flatten())
        
        if polished_val > best_val:
            best_val = polished_val
            best_bary = polished_bary
            
        # If we have a very good score, we might stop early?
        # The theoretical max for 11 points in a triangle is around 1/30 to 1/40 of the triangle area?
        # Area of equilateral triangle with side 1 is sqrt(3)/4 ~ 0.433.
        # If min area is ~ 0.01, that's a good score.
        # Let's just keep going until time runs out or we've tried all inits.
        
    # If no best_bary found (should not happen), use the first init
    if best_bary is None:
        best_bary = inits[0]
        
    # Final check
    cart = bary_to_cart(best_bary)
    return cart

# EVOLVE-BLOCK-END
