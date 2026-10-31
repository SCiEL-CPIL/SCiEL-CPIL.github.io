import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import time

N = 11

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
# Barycentric coordinates: a = 1 - x - y/height, b = x, c = y/height
# where height = sqrt(3)/2
H = np.sqrt(3) / 2.0

def barycentric_to_cartesian(bary):
    """Convert (N,3) barycentric to (N,2) cartesian, ensuring inside triangle."""
    # bary: (N, 3) with columns a, b, c where a+b+c=1, all >= 0
    # x = b + 0.5*c, y = H*c
    b = bary[:, 1]
    c = bary[:, 2]
    x = b + 0.5 * c
    y = H * c
    return np.column_stack([x, y])

def cartesian_to_barycentric(cart):
    """Convert (N,2) cartesian to (N,3) barycentric."""
    x = cart[:, 0]
    y = cart[:, 1]
    c = y / H
    b = x - 0.5 * c
    a = 1.0 - b - c
    return np.column_stack([a, b, c])

def min_area_from_bary(bary):
    """Compute minimum triangle area from barycentric coordinates."""
    cart = barycentric_to_cartesian(bary)
    p1 = cart[tri_idx[:, 0]]
    p2 = cart[tri_idx[:, 1]]
    p3 = cart[tri_idx[:, 2]]
    # Area = 0.5 * |cross(p2-p1, p3-p1)|
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def soft_min_area(bary, beta=200.0):
    """Smooth surrogate for min area using log-sum-exp."""
    cart = barycentric_to_cartesian(bary)
    p1 = cart[tri_idx[:, 0]]
    p2 = cart[tri_idx[:, 1]]
    p3 = cart[tri_idx[:, 2]]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return -log_sum / beta

def objective_bary(bary_flat):
    """Objective to minimize (negative of soft min area)."""
    bary = bary_flat.reshape(N, 3)
    # Project to valid barycentric: clip negatives to 0, renormalize
    bary_clipped = np.clip(bary, 0, None)
    row_sums = bary_clipped.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    bary_norm = bary_clipped / row_sums
    return -soft_min_area(bary_norm)

def objective_bary_hard(bary_flat):
    """Objective using hard min (for SLSQP final polish)."""
    bary = bary_flat.reshape(N, 3)
    bary_clipped = np.clip(bary, 0, None)
    row_sums = bary_clipped.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    bary_norm = bary_clipped / row_sums
    return -min_area_from_bary(bary_norm)

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    # Initial best: seed-like random points
    rng = np.random.default_rng(42)
    best_bary = None
    best_val = -np.inf
    
    # Generate initial population for DE
    def random_bary():
        # Random points inside triangle using rejection in barycentric
        u = rng.random(N)
        v = rng.random(N)
        # Map to barycentric: a = 1-u-v, b = u, c = v (if u+v<=1)
        # Use a different mapping for uniform distribution
        # Actually for uniform in triangle: a=1-sqrt(u), b=sqrt(u)*v, c=sqrt(u)*(1-v)
        s = np.sqrt(rng.random(N))
        t = rng.random(N)
        a = 1 - s
        b = s * t
        c = s * (1 - t)
        bary = np.column_stack([a, b, c])
        return bary.flatten()
    
    # First, try a few random starts with local polish
    for trial in range(5):
        if time.time() > deadline - 30:
            break
        bary0 = random_bary()
        val = objective_bary(bary0)
        if -val > best_val:
            best_val = -val
            best_bary = bary0.copy()
    
    # Differential Evolution global search
    if time.time() < deadline - 45:
        try:
            bounds = [(0, 1)] * (N * 3)
            de_result = differential_evolution(
                objective_bary,
                bounds=bounds,
                seed=42,
                maxiter=80,
                tol=1e-6,
                popsize=15,
                mutation=(0.3, 1.0),
                recombination=0.7,
                polish=False,
                init='sobol',
                updating='immediate',
                callback=None
            )
            if -de_result.fun > best_val:
                best_val = -de_result.fun
                best_bary = de_result.x.copy()
        except Exception:
            pass
    
    # Local polish with SLSQP using soft objective
    if time.time() < deadline - 20:
        try:
            # Multiple SLSQP restarts from best and nearby points
            slsqp_starts = [best_bary.copy()]
            # Add a few perturbed starts
            for i in range(3):
                perturbed = best_bary + rng.normal(0, 0.05, size=best_bary.shape)
                slsqp_starts.append(perturbed)
            
            for s in slsqp_starts:
                if time.time() > deadline - 10:
                    break
                result = minimize(
                    objective_bary,
                    s,
                    method='SLSQP',
                    bounds=[(0, 1)] * (N * 3),
                    options={'maxiter': 200, 'ftol': 1e-12}
                )
                if -result.fun > best_val:
                    best_val = -result.fun
                    best_bary = result.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish with hard min objective
    if time.time() < deadline - 5:
        try:
            result = minimize(
                objective_bary_hard,
                best_bary,
                method='SLSQP',
                bounds=[(0, 1)] * (N * 3),
                options={'maxiter': 300, 'ftol': 1e-14}
            )
            if -result.fun > best_val:
                best_val = -result.fun
                best_bary = result.x.copy()
        except Exception:
            pass
        
        # One more Nelder-Mead polish
        if time.time() < deadline - 2:
            try:
                result = minimize(
                    objective_bary_hard,
                    best_bary,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-14}
                )
                if -result.fun > best_val:
                    best_val = -result.fun
                    best_bary = result.x.copy()
            except Exception:
                pass
    
    # Convert to cartesian
    bary_final = best_bary.reshape(N, 3)
    bary_final = np.clip(bary_final, 0, None)
    row_sums = bary_final.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    bary_final = bary_final / row_sums
    cart = barycentric_to_cartesian(bary_final)
    
    return cart

# EVOLVE-BLOCK-END
