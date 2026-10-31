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

def soft_min_area(bary, beta=500.0):
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

def objective_bary(bary_flat, beta=500.0):
    """Objective to minimize (negative of soft min area)."""
    bary = bary_flat.reshape(N, 3)
    # Project to valid barycentric: clip negatives to 0, renormalize
    bary_clipped = np.clip(bary, 0, None)
    row_sums = bary_clipped.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    bary_norm = bary_clipped / row_sums
    return -soft_min_area(bary_norm, beta)

def objective_bary_hard(bary_flat):
    """Objective using hard min (for final polish)."""
    bary = bary_flat.reshape(N, 3)
    bary_clipped = np.clip(bary, 0, None)
    row_sums = bary_clipped.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    bary_norm = bary_clipped / row_sums
    return -min_area_from_bary(bary_norm)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Initial best: structured guess
    rng = np.random.default_rng(42)
    best_bary = None
    best_val = -np.inf
    
    # Generate a structured initial guess (grid-like)
    # For N=11, a 3x3 grid plus corners? Or just random?
    # Let's use a few random starts and a grid start.
    
    def random_bary():
        s = np.sqrt(rng.random(N))
        t = rng.random(N)
        a = 1 - s
        b = s * t
        c = s * (1 - t)
        bary = np.column_stack([a, b, c])
        return bary.flatten()

    def grid_bary():
        # Simple grid in barycentric coordinates
        # This is not perfectly uniform but provides structure
        barys = []
        # Add corners
        barys.append([1, 0, 0])
        barys.append([0, 1, 0])
        barys.append([0, 0, 1])
        # Add midpoints
        barys.append([0.5, 0.5, 0])
        barys.append([0.5, 0, 0.5])
        barys.append([0, 0.5, 0.5])
        # Add center
        barys.append([1/3, 1/3, 1/3])
        # Add 4 more points to reach 11
        # Place them near the center or edges
        # For now, just fill remaining with random close to center
        for _ in range(N - len(barys)):
            s = np.sqrt(rng.random())
            t = rng.random()
            a = 1 - s
            b = s * t
            c = s * (1 - t)
            # Pull towards center (1/3, 1/3, 1/3)
            a = 0.7 * a + 0.3 * (1/3)
            b = 0.7 * b + 0.3 * (1/3)
            c = 0.7 * c + 0.3 * (1/3)
            # Normalize
            s_sum = a + b + c
            a, b, c = a/s_sum, b/s_sum, c/s_sum
            barys.append([a, b, c])
        
        bary = np.array(barys[:N])
        return bary.flatten()

    # Evaluate initial guesses
    for init_func in [random_bary, grid_bary]:
        for _ in range(5):
            if time.time() > deadline - 60: break
            bary0 = init_func()
            val = objective_bary(bary0)
            if -val > best_val:
                best_val = -val
                best_bary = bary0.copy()
    
    # Differential Evolution global search
    # Reduce iterations to save time for polishing
    if time.time() < deadline - 50:
        try:
            bounds = [(0, 1)] * (N * 3)
            de_result = differential_evolution(
                lambda x: objective_bary(x, beta=100.0), # Lower beta for smoother global landscape
                bounds=bounds,
                seed=42,
                maxiter=30,
                tol=1e-4,
                popsize=10,
                mutation=(0.3, 1.0),
                recombination=0.7,
                polish=False,
                init='sobol',
                updating='immediate'
            )
            if -de_result.fun > best_val:
                # Convert back to standard objective value
                # Note: DE minimized -soft_min(beta=100). 
                # We need to evaluate with the best beta or hard min to compare fairly?
                # Actually, we just want the location.
                val_hard = objective_bary_hard(de_result.x)
                if -val_hard > best_val:
                    best_val = -val_hard
                    best_bary = de_result.x.copy()
        except Exception:
            pass
    
    # Local Polish Loop
    # Alternate between SLSQP (soft) and Nelder-Mead (hard)
    polish_iterations = 10
    for i in range(polish_iterations):
        if time.time() > deadline - 10:
            break
        
        # 1. SLSQP with Soft Min (Higher Beta for precision)
        if time.time() < deadline - 15:
            try:
                result = minimize(
                    lambda x: objective_bary(x, beta=1000.0),
                    best_bary,
                    method='SLSQP',
                    bounds=[(0, 1)] * (N * 3),
                    options={'maxiter': 100, 'ftol': 1e-12}
                )
                val_hard = objective_bary_hard(result.x)
                if -val_hard > best_val:
                    best_val = -val_hard
                    best_bary = result.x.copy()
            except Exception:
                pass
        
        # 2. Nelder-Mead with Hard Min (Derivative-free, robust)
        if time.time() < deadline - 5:
            try:
                result = minimize(
                    objective_bary_hard,
                    best_bary,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-14, 'adaptive': True}
                )
                val_hard = objective_bary_hard(result.x)
                if -val_hard > best_val:
                    best_val = -val_hard
                    best_bary = result.x.copy()
            except Exception:
                pass

    # Final high-precision polish
    if time.time() < deadline - 2:
        try:
            result = minimize(
                objective_bary_hard,
                best_bary,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
            )
            val_hard = objective_bary_hard(result.x)
            if -val_hard > best_val:
                best_val = -val_hard
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
