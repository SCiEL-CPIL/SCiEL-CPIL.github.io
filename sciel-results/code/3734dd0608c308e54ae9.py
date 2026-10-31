import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import time

N = 11

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
# Barycentric coordinates: u, v, w where u+v+w=1
# Point = u*A + v*B + w*C
# A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
# x = v + 0.5*w
# y = (sqrt(3)/2)*w
H = np.sqrt(3) / 2.0

def bary_to_cart(bary):
    """
    Convert barycentric coords (u, v) to Cartesian (x, y).
    bary: (N, 2) array of (u, v). w = 1 - u - v.
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
    Returns array of shape (165,) containing the area ratios (relative to triangle area).
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
    # Area = 0.5 * |det|, but we are maximizing min area, so we can just use |det|
    # The relative area to the big triangle (area 1 in barycentric determinant units? No, big triangle det is 1)
    # Det of (0,0), (1,0), (0,1) in barycentric is 1.
    # So |det| is the ratio of the small triangle area to the big triangle area.
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    return np.abs(det)

def clamp_bary(bary, eps=1e-9):
    """
    Clamp barycentric coordinates to the valid triangle region.
    u >= 0, v >= 0, u+v <= 1
    """
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    # Clip to valid range
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)
    
    # Ensure u + v <= 1
    sums = u + v
    invalid = sums > 1.0
    if np.any(invalid):
        scale = 1.0 / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    return np.column_stack([u, v])

def soft_min_area(bary, beta=200.0):
    """
    Smooth surrogate for min area using log-sum-exp.
    """
    areas = calc_areas_bary(bary)
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    # Handle potential overflow/underflow, though areas are small
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return -log_sum / beta

def objective_soft(bary_flat, beta=200.0):
    """
    Objective to minimize (negative of soft min area).
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    return -soft_min_area(bary, beta)

def objective_hard(bary_flat):
    """
    Objective using hard min (for final polish).
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration.
    For 11 points, we can use a triangular arrangement.
    """
    best_bary = None
    best_val = -np.inf
    
    # Try different spacings
    for s in np.linspace(0.1, 0.5, 20):
        points = []
        for i in range(5):
            for j in range(5):
                u = i * s
                v = j * s
                if u + v <= 1.0 and len(points) < N:
                    points.append([u, v])
        
        if len(points) == N:
            bary = np.array(points)
            val = -objective_hard(bary.flatten())
            if val > best_val:
                best_val = val
                best_bary = bary.copy()
    
    if best_bary is None:
        # Fallback: random
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
        
    return best_bary

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    # 1. Initial Best from Hexagonal Lattice
    hex_init = generate_hexagonal_init()
    best_bary = hex_init.copy()
    best_val = -objective_hard(best_bary.flatten())
    
    # 2. Differential Evolution for Global Search
    # DE is robust to non-smoothness if we use a soft objective with moderate beta
    if time.time() < deadline - 60:
        try:
            bounds = [(0, 1)] * (N * 2)
            # Use a soft objective with beta=100 for DE to guide it towards good basins
            de_result = differential_evolution(
                lambda x: objective_soft(x, beta=100.0),
                bounds=bounds,
                seed=42,
                maxiter=50,
                tol=1e-5,
                popsize=15,
                mutation=(0.3, 1.0),
                recombination=0.7,
                polish=False,
                init='sobol',
                updating='immediate'
            )
            
            de_bary = de_result.x.reshape(N, 2)
            de_val = -objective_hard(de_bary.flatten())
            
            if de_val > best_val:
                best_val = de_val
                best_bary = de_bary.copy()
        except Exception:
            pass

    # 3. Local Polish Phase
    # We will run multiple local optimizations starting from:
    # a) The current best
    # b) A few random perturbations of the best
    # c) The hexagonal init (in case DE went astray)
    
    starts = [best_bary.copy(), hex_init.copy()]
    rng = np.random.default_rng(123)
    for _ in range(4):
        perturbed = best_bary + rng.normal(0, 0.05, best_bary.shape)
        perturbed = clamp_bary(perturbed)
        starts.append(perturbed)
    
    for s in starts:
        if time.time() > deadline - 15:
            break
        
        try:
            # Stage 1: Soft min with medium beta (Powell is derivative-free and robust)
            res1 = minimize(
                lambda x: objective_soft(x, beta=100.0),
                s.flatten(),
                method='Powell',
                options={'maxiter': 200, 'xtol': 1e-8, 'ftol': 1e-8}
            )
            bary1 = clamp_bary(res1.x.reshape(N, 2))
            
            # Stage 2: Soft min with high beta (sharper landscape)
            res2 = minimize(
                lambda x: objective_soft(x, beta=500.0),
                bary1.flatten(),
                method='Powell',
                options={'maxiter': 200, 'xtol': 1e-10, 'ftol': 1e-10}
            )
            bary2 = clamp_bary(res2.x.reshape(N, 2))
            
            # Stage 3: Hard min (Nelder-Mead for final touch)
            res3 = minimize(
                objective_hard,
                bary2.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            bary3 = clamp_bary(res3.x.reshape(N, 2))
            
            val3 = -objective_hard(bary3.flatten())
            if val3 > best_val:
                best_val = val3
                best_bary = bary3.copy()
                
        except Exception:
            pass

    # 4. Final Verification and Conversion
    best_bary = clamp_bary(best_bary)
    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
