"""
Strategy: Direct smooth optimization of a repulsive force objective.
Instead of binary searching for a feasible radius r and minimizing violations (which has flat zero-loss regions),
we directly optimize a smooth objective function that penalizes close pairs and wall violations.
Objective: minimize sum_{i<j} (1 / max(d_ij^2, eps)^2) + sum_i (1 / max(wall_i^2, eps)^2)
This is differentiable everywhere (with eps) and guides the optimizer to increase the minimum distance.
We also use a coordinate transformation to map unbounded optimization variables to the interior of the square [-0.5, 0.5],
ensuring the circles stay inside the square without hard boundary constraints.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def _score(cents):
    """Compute the packing radius (the score)."""
    n = len(cents)
    if n == 0:
        return 0.0
    # Wall distances
    wall_min = np.min(np.minimum(0.5 - np.abs(cents[:, 0]), 0.5 - np.abs(cents[:, 1])))
    # Pairwise distances
    if n > 1:
        diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_min = np.min(dists) / 2.0
    else:
        pair_min = 0.5
    return min(wall_min, pair_min)

def _map_to_square(u):
    """
    Map unbounded variables u to the interior of the square [-0.5, 0.5].
    We use a scaled tanh to ensure we don't hit the boundaries exactly, 
    but we want to allow them to get close.
    Actually, a simple linear map from a bounded box is better if we use bounds.
    However, to avoid boundary issues, let's use a soft mapping.
    Let's stick to standard bounds in L-BFGS-B but use the smooth objective.
    The smooth objective is:
    f(x) = sum_{i<j} (2r_target / d_ij)^4 + sum_i (r_target / wall_i)^4
    Wait, we don't know r_target. 
    A common trick is to minimize the sum of inverse distances.
    f(x) = sum_{i<j} 1/d_ij^2 + sum_i 1/wall_i^2
    Minimizing this tends to maximize the minimum distance.
    """
    return u

def _smooth_objective(x, alpha=4.0, beta=4.0):
    """
    x: shape (2N,)
    Returns a smooth scalar value to minimize.
    We want to maximize the minimum distance.
    Minimizing sum of (1/d^p) approximates maximizing min(d).
    We add a term for walls as well.
    """
    c = x.reshape(N, 2)
    
    # Wall distances: distance to nearest wall
    # x in [-0.5, 0.5]. Wall distance for x_i is 0.5 - |x_i|
    wall_x = 0.5 - np.abs(c[:, 0])
    wall_y = 0.5 - np.abs(c[:, 1])
    
    # To avoid division by zero, add a small epsilon
    eps = 1e-6
    wall_x = np.maximum(wall_x, eps)
    wall_y = np.maximum(wall_y, eps)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    # We want to penalize small distances.
    # Using 1/d^2 is standard.
    # We can weight the wall and pair terms.
    # If we just minimize sum(1/d^2), the circles will spread out.
    
    # To make the objective more sensitive to the MINIMUM distance,
    # we can use a higher power, e.g., 1/d^4.
    # This makes the optimizer focus more on the closest pairs.
    
    pair_inv_sq = np.sum(1.0 / dists_sq)
    wall_inv_sq = np.sum(1.0 / wall_x**2) + np.sum(1.0 / wall_y**2)
    
    return pair_inv_sq + wall_inv_sq

def _gradient_smooth_objective(x, alpha=4.0, beta=4.0):
    """
    Gradient of the smooth objective.
    f(x) = sum_{i<j} 1/d_ij^2 + sum_i (1/wall_x_i^2 + 1/wall_y_i^2)
    
    d/dx_i (1/d_ij^2) = d/dx_i ((x_i - x_j)^2 + (y_i - y_j)^2)^-1
    = -1 * ((x_i - x_j)^2 + (y_i - y_j)^2)^-2 * 2(x_i - x_j)
    = -2(x_i - x_j) / d_ij^4
    
    Summing over j != i:
    grad_i = sum_{j != i} -2(x_i - x_j) / d_ij^4
    
    For walls:
    d/dx_i (1/wall_x_i^2) where wall_x_i = 0.5 - |x_i|
    If x_i > 0, wall = 0.5 - x_i, d/wall = -1.
    d/dx_i (1/wall^2) = -1/wall^3 * (-1) = 1/wall^3
    If x_i < 0, wall = 0.5 + x_i, d/wall = 1.
    d/dx_i (1/wall^2) = -1/wall^3 * 1 = -1/wall^3
    So grad = sign(x_i) / wall_i^3
    """
    c = x.reshape(N, 2)
    grad = np.zeros_like(c)
    
    # Pairwise gradient
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, 1e-10) # Avoid div by zero
    
    # 1/d^4 term
    # Note: dists_sq is d^2. So 1/d^4 is 1/dists_sq^2.
    inv_d4 = 1.0 / (dists_sq**2)
    
    # grad_i = sum_j -2 * diff_ij * inv_d4_ij
    # diff_ij = c_i - c_j
    grad_pairs = -2.0 * np.sum(diff * inv_d4[:, :, np.newaxis], axis=1)
    
    # Wall gradient
    wall_x = 0.5 - np.abs(c[:, 0])
    wall_y = 0.5 - np.abs(c[:, 1])
    wall_x = np.maximum(wall_x, 1e-6)
    wall_y = np.maximum(wall_y, 1e-6)
    
    # sign(x) / wall^3
    sign_x = np.sign(c[:, 0])
    sign_y = np.sign(c[:, 1])
    
    grad[:, 0] += grad_pairs[:, 0] + sign_x / (wall_x**3)
    grad[:, 1] += grad_pairs[:, 1] + sign_y / (wall_y**3)
    
    return grad.flatten()

def _construct_hexagonal():
    """Create a hexagonal packing arrangement for 50 circles."""
    best_cents = None
    best_r = 0

    for n_rows in [5, 6, 7, 8]:
        base = N // n_rows
        extra = N % n_rows
        row_counts = [base + (1 if i < extra else 0) for i in range(n_rows)]

        max_rc = max(row_counts)
        r_horiz = 1.0 / (2.0 * max_rc)
        r_vert = 1.0 / (2.0 + (n_rows - 1) * np.sqrt(3.0))
        r = min(r_horiz, r_vert)

        if r <= best_r:
            continue

        centers = []
        total_y = (n_rows - 1) * 2 * r * np.sqrt(3.0) / 2.0
        y_start = -0.5 + r + total_y / 2.0

        for row_idx in range(n_rows):
            rc = row_counts[row_idx]
            y = y_start + row_idx * 2 * r * np.sqrt(3.0) / 2.0
            x_span = (rc - 1) * 2 * r
            x_start = -0.5 + r + x_span / 2.0
            offset = (2 * r * np.sqrt(3.0) / 2.0) * 0.5 * (row_idx % 2)
            for col in range(rc):
                x = x_start + col * 2 * r + offset
                centers.append([x, y])

        centers = np.array(centers[:N])
        r_actual = _score(centers)

        if r_actual > best_r:
            best_r = r_actual
            best_cents = centers.copy()

    if best_cents is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_cents = grid[:N]
        best_r = _score(best_cents)

    return best_cents, best_r

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    # Step 1: Hexagonal initial arrangement
    cents, r0 = _construct_hexagonal()
    best_cents = cents.copy()
    best_r = r0
    
    x0 = cents.reshape(-1)
    
    # Step 2: Direct smooth optimization
    # We use L-BFGS-B with bounds to keep points in the square.
    # The objective is smooth, so this should converge well.
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # We might run multiple optimizations with different starting points or parameters
    # to avoid local minima.
    
    # First, run from the hexagonal start
    try:
        result = minimize(
            _smooth_objective, 
            x0, 
            jac=_gradient_smooth_objective,
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
        
        elapsed = time.time() - start_time
        if elapsed < deadline - 10:
            x_opt = result.x
            cents_opt = x_opt.reshape(N, 2)
            r_opt = _score(cents_opt)
            
            if r_opt > best_r:
                best_r = r_opt
                best_cents = cents_opt
    except Exception as e:
        pass

    # If we have time, try a few random restarts or perturbations
    # to see if we can find a better local minimum.
    remaining = deadline - (time.time() - start_time)
    if remaining > 20:
        # Try a few random perturbations of the best solution
        for i in range(3):
            if time.time() - start_time > deadline - 5:
                break
            
            # Perturb the best solution slightly
            noise = np.random.normal(0, 0.01, size=(N, 2))
            x_pert = best_cents.reshape(-1) + noise.reshape(-1)
            x_pert = np.clip(x_pert, -0.49, 0.49) # Keep strictly inside
            
            try:
                result = minimize(
                    _smooth_objective, 
                    x_pert, 
                    jac=_gradient_smooth_objective,
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
                
                x_opt = result.x
                cents_opt = x_opt.reshape(N, 2)
                r_opt = _score(cents_opt)
                
                if r_opt > best_r:
                    best_r = r_opt
                    best_cents = cents_opt
            except Exception as e:
                pass

    # Final safety check: ensure all points are within bounds
    best_cents = np.clip(best_cents, -0.5, 0.5)
    
    return best_cents
# EVOLVE-BLOCK-END
