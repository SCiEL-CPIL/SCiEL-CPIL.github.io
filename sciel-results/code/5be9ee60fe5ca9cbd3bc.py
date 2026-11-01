"""
Strategy: Direct smooth optimization of a packing proxy objective.
Instead of binary searching for a feasible radius r and minimizing violations (which has flat zero-loss regions),
we directly optimize a smooth objective function that penalizes close pairs and wall violations.
Objective: minimize sum_{i<j} (1 / max(d_ij^2, eps)^2) + sum_i (1 / max(wall_i^2, eps)^2)
This is differentiable everywhere (with eps) and guides the optimizer to increase the minimum distance.
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

def _objective_and_grad(x):
    """
    Objective: Minimize sum of inverse squared distances (pairwise and wall).
    This is a smooth proxy for maximizing the minimum distance.
    x is flat array of shape (2N,).
    """
    c = x.reshape(N, 2)
    eps = 1e-10
    
    # Pairwise term
    # d_ij^2 = (x_i - x_j)^2 + (y_i - y_j)^2
    # term = 1 / (d_ij^2 + eps)^2
    # grad w.r.t x_i: -2 * (x_i - x_j) * term / (d_ij^2 + eps)
    
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    d_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(d_sq, 1.0) # Avoid div by zero, will mask later
    
    # We want to minimize sum_{i<j} 1/(d_ij^2 + eps)^2
    # Let u_ij = d_ij^2 + eps
    # f = sum 1/u_ij^2
    # df/du = -2/u^3
    # du/dx_i = 2*(x_i - x_j)
    # df/dx_i = sum_j -2/u_ij^3 * 2*(x_i - x_j)
    
    u = d_sq + eps
    inv_u2 = 1.0 / (u**2)
    # Mask diagonal
    inv_u2[np.diag_indices(N)] = 0.0
    
    # Gradient for pairwise part
    # grad_x_i = sum_{j != i} -4 * (x_i - x_j) / u_ij^3
    # We can compute this via matrix multiplication or loops.
    # Let's use a trick: 
    # grad[i] = sum_j weight_ij * (c[i] - c[j])
    # weight_ij = -4 / u_ij^3
    weights = -4.0 / (u**3)
    weights[np.diag_indices(N)] = 0.0
    
    # grad_c = weights @ c - weights.sum(axis=1, keepdims=True) * c
    # Wait, sum_j w_ij (c_i - c_j) = c_i * sum_j w_ij - sum_j w_ij c_j
    sum_weights = weights.sum(axis=1)
    weighted_c = weights @ c
    
    grad_c = c * sum_weights[:, np.newaxis] - weighted_c
    
    # Wall term
    # distance to wall for point i: d_i = min(0.5 - |x_i|, 0.5 - |y_i|)
    # This min is non-differentiable. 
    # Instead, let's penalize all 4 walls smoothly?
    # Or just use the closest wall?
    # To keep it smooth and simple, let's penalize penetration into the "forbidden zone" outside [-(0.5-r), 0.5-r]?
    # No, we don't have r.
    # Let's use a soft wall penalty:
    # For each coord, penalty = 1 / (0.5 - |coord| + eps)^2 ?
    # This pushes points away from the boundary.
    
    # Let's define wall distance for each point as min(0.5-|x|, 0.5-|y|).
    # To make it differentiable, we can approximate min(a,b) with -log(exp(-a/t)+exp(-b/t))*t for small t,
    # or just penalize both x and y walls separately.
    # Penalizing both is safer and symmetric.
    
    # wall_x_dist = 0.5 - |x|
    # wall_y_dist = 0.5 - |y|
    # obj_wall = sum_i (1/(wall_x_dist_i + eps)^2 + 1/(wall_y_dist_i + eps)^2)
    
    wx = 0.5 - np.abs(c[:, 0])
    wy = 0.5 - np.abs(c[:, 1])
    
    # Clip to avoid huge values if outside, though bounds should keep it inside.
    # But if x > 0.5, wx < 0. We want to push it back.
    # If we use 1/wx^2, and wx is negative, it's still positive but gradient direction?
    # d/dx (1/(0.5-|x|)^2). If x>0, d/dx (0.5-x)^-2 = -2(0.5-x)^-3 * (-1) = 2/(0.5-x)^3.
    # If x > 0.5, 0.5-x is negative, so gradient is negative, pushing x down. Correct.
    
    obj_wall = np.sum(1.0 / (wx + eps)**2) + np.sum(1.0 / (wy + eps)**2)
    
    # Gradient for wall part
    # d/dx_i 1/(0.5-|x_i|)^2
    # If x_i > 0: d/dx (0.5-x)^-2 = 2(0.5-x)^-3
    # If x_i < 0: d/dx (0.5+x)^-2 = -2(0.5+x)^-3
    # Generally: 2 * sign(x_i) / (0.5-|x_i|)^3 ? 
    # Let's check: x>0, sign=1, 2/(0.5-x)^3. Correct.
    # x<0, sign=-1, -2/(0.5+x)^3. Correct.
    
    grad_wx = 2.0 * np.sign(c[:, 0]) / (wx + eps)**3
    grad_wy = 2.0 * np.sign(c[:, 1]) / (wy + eps)**3
    
    grad_c[:, 0] += grad_wx
    grad_c[:, 1] += grad_wy
    
    obj_pair = np.sum(inv_u2)
    total_obj = obj_pair + obj_wall
    
    total_grad = grad_c.flatten()
    
    return total_obj, total_grad

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
    deadline = 145.0  # Safety margin before 150s kill
    
    # Step 1: Hexagonal initial arrangement
    cents, r0 = _construct_hexagonal()
    best_cents = cents.copy()
    best_r = r0
    
    # Step 2: Multi-restart optimization
    # We will try a few different initializations and run L-BFGS-B on the smooth objective.
    
    inits = []
    # 1. Hexagonal
    inits.append(cents.copy())
    
    # 2. Perturbed Hexagonal
    rng = np.random.default_rng(42)
    for _ in range(5):
        pert = cents + rng.normal(0, 0.01, size=cents.shape)
        # Clip to bounds
        pert = np.clip(pert, -0.49, 0.49)
        inits.append(pert)
        
    # 3. Random uniform
    for _ in range(3):
        rand_cents = rng.uniform(-0.45, 0.45, size=(N, 2))
        inits.append(rand_cents)
        
    # 4. Grid
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    inits.append(grid[:N])
    
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    for i, init_cents in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        x0 = init_cents.reshape(-1)
        
        try:
            result = minimize(
                _objective_and_grad, 
                x0, 
                jac=True, # We return (f, g)
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            x_opt = result.x
            cents_opt = x_opt.reshape(N, 2)
            
            # Ensure valid
            if not np.all(np.isfinite(cents_opt)):
                continue
                
            # Clip to be safe
            cents_opt = np.clip(cents_opt, -0.5, 0.5)
            
            r_opt = _score(cents_opt)
            
            if r_opt > best_r:
                best_r = r_opt
                best_cents = cents_opt.copy()
                
        except Exception as e:
            continue
            
    # Final safety check
    best_cents = np.clip(best_cents, -0.5, 0.5)
    
    return best_cents
# EVOLVE-BLOCK-END
