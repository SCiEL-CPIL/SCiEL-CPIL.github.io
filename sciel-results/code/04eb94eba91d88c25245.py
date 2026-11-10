"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def _compute_radius(centers):
    """Compute the largest common radius for given centers."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    if centers.shape != (N, 3):
        return 0.0
    
    # Distance from each center to nearest wall
    # Walls are at -0.5 and 0.5
    wall_dists = np.min(np.array([
        centers[:, 0] + 0.5,
        -centers[:, 0] + 0.5,
        centers[:, 1] + 0.5,
        -centers[:, 1] + 0.5,
        centers[:, 2] + 0.5,
        -centers[:, 2] + 0.5,
    ]), axis=1)
    
    if np.any(wall_dists < -1e-10):
        return 0.0
    
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    r = 0.5 * min(min_pair, min_wall)
    return r if r > 0 else 0.0

def _softmin_grad(vals, beta):
    """
    Compute soft-min of vals and its gradient.
    softmin(vals, beta) = -1/beta * log(sum(exp(-beta * vals)))
    grad_i = exp(-beta * vals_i) / sum(exp(-beta * vals))
    Note: This is the gradient of the soft-min w.r.t vals.
    """
    # Shift for numerical stability
    vals_shifted = vals - np.min(vals)
    exp_vals = np.exp(-beta * vals_shifted)
    sum_exp = np.sum(exp_vals)
    # Soft-min value
    soft_min_val = -1.0 / beta * np.log(sum_exp) + np.min(vals)
    # Gradient w.r.t vals
    grad = exp_vals / sum_exp
    return soft_min_val, grad

def construct():
    """Construct centers for N equal spheres in unit cube."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s/150s limit
    
    # 1. Initial Grid Placement
    # Arrange N spheres in a grid. 
    # For N=100, a 5x5x4 grid is 100.
    # Or 4x5x5.
    # Let's try to find a grid shape close to cube.
    # 4*5*5 = 100.
    m1, m2, m3 = 4, 5, 5
    ticks1 = (np.arange(m1) + 0.5) / m1 - 0.5
    ticks2 = (np.arange(m2) + 0.5) / m2 - 0.5
    ticks3 = (np.arange(m3) + 0.5) / m3 - 0.5
    
    grid = np.array([(x, y, z) for z in ticks3 for y in ticks2 for x in ticks1])
    best_centers = grid[:N].copy()
    best_radius = _compute_radius(best_centers)
    
    # Optimization Function
    def objective_and_grad(x_flat, beta):
        centers = x_flat.reshape(N, 3)
        
        # Pairwise distances
        dists = cdist(centers, centers)
        np.fill_diagonal(dists, np.inf)
        
        # Wall distances
        wall_dists = np.min(np.array([
            centers[:, 0] + 0.5,
            -centers[:, 0] + 0.5,
            centers[:, 1] + 0.5,
            -centers[:, 1] + 0.5,
            centers[:, 2] + 0.5,
            -centers[:, 2] + 0.5,
        ]), axis=1)
        
        # We want to maximize the minimum of (dists/2) and wall_dists.
        # Let's collect all these values into a single vector V.
        # V will contain N*(N-1)/2 pair distances / 2 and N wall distances.
        
        # To avoid huge memory usage for gradient, we compute it carefully.
        # However, N=100 is small enough.
        
        # Get unique pairs i < j
        i_idx, j_idx = np.triu_indices(N, k=1)
        pair_dists = dists[i_idx, j_idx] / 2.0
        
        # Combine pair distances and wall distances
        # Note: Wall distances are per-sphere, pair distances are per-pair.
        # The objective is min( min(pair_dists), min(wall_dists) )
        # We can treat them as a single pool of values for the soft-min.
        
        all_vals = np.concatenate([pair_dists, wall_dists])
        
        # Compute soft-min
        val, grad_all = _softmin_grad(all_vals, beta)
        
        # Objective is -val (minimize -softmin)
        obj = -val
        
        # Gradient w.r.t x
        grad_flat = np.zeros_like(x_flat)
        
        # Gradient from pair distances
        # grad_all[0 : len(pair_dists)] corresponds to pair_dists
        grad_pair = grad_all[:len(pair_dists)]
        
        # For each pair (i, j), distance d_ij = ||c_i - c_j|| / 2
        # d/d c_i (d_ij) = (c_i - c_j) / (2 * ||c_i - c_j||) * 2 ? No.
        # d_ij = 0.5 * ||c_i - c_j||
        # grad_{c_i} d_ij = 0.5 * (c_i - c_j) / ||c_i - c_j||
        # grad_{c_j} d_ij = 0.5 * (c_j - c_i) / ||c_i - c_j||
        
        # We need to accumulate these gradients.
        # Using numpy advanced indexing and bincount or similar is tricky for vectors.
        # We can loop over pairs if N is small, but N=100 -> ~5000 pairs. Looping in Python is slow.
        # Let's vectorize the accumulation.
        
        # Diff vector for each pair
        diffs = centers[i_idx] - centers[j_idx]
        norms = np.linalg.norm(diffs, axis=1)
        norms[norms == 0] = 1e-8 # Avoid div by zero
        
        # Unit vectors
        unit_vecs = diffs / norms[:, np.newaxis]
        
        # Gradient contribution for each pair k
        # g_k = grad_pair[k] * 0.5 * unit_vec[k]
        # Add to i_idx[k], subtract from j_idx[k]
        
        # This requires a scatter-add operation.
        # grad_flat[i_idx] += grad_pair[:, None] * 0.5 * unit_vecs
        # grad_flat[j_idx] -= grad_pair[:, None] * 0.5 * unit_vecs
        
        # Numpy doesn't have built-in scatter-add for arrays in older versions easily without np.add.at
        np.add.at(grad_flat, i_idx, grad_pair[:, None] * 0.5 * unit_vecs)
        np.add.at(grad_flat, j_idx, -grad_pair[:, None] * 0.5 * unit_vecs)
        
        # Gradient from wall distances
        # grad_all[len(pair_dists):] corresponds to wall_dists
        grad_wall = grad_all[len(pair_dists):]
        
        # wall_dist_i = min(c_i+0.5, -c_i+0.5, ...)
        # The gradient of the min is the gradient of the active constraint.
        # But soft-min mixes them.
        # Wait, my `all_vals` just concatenated them.
        # The soft-min gradient `grad_wall` gives the weight of each wall distance in the soft-min.
        # But `wall_dists` itself is a min of 6 values.
        # So we have nested mins?
        # Actually, `wall_dists` is already the min distance to any wall.
        # So `all_vals` contains the "effective" distance for each sphere.
        # The gradient of `wall_dists[i]` w.r.t `c_i` is:
        # If the active wall is x+0.5, grad is (1,0,0).
        # If x-0.5, grad is (-1,0,0).
        # Etc.
        
        # We need to know which wall is "active" for each sphere to get the gradient direction.
        # Let's re-calculate wall gradients explicitly.
        
        # For each sphere i, the distance to 6 walls.
        # w_0 = c_x + 0.5, w_1 = -c_x + 0.5, ...
        # w_min = min(w_0..w_5)
        # grad w_min w.r.t c_x is 1 if w_0 is min, -1 if w_1 is min, 0 otherwise.
        
        # Let's compute the "active" direction for each sphere.
        # This is effectively the gradient of the min function.
        # Since we are using soft-min on the *pool* of values, we need the gradient of `wall_dists` w.r.t `centers`.
        
        # Let's compute wall gradients properly.
        # Create a matrix of wall distances W (N x 6)
        W = np.zeros((N, 6))
        W[:, 0] = centers[:, 0] + 0.5
        W[:, 1] = -centers[:, 0] + 0.5
        W[:, 2] = centers[:, 1] + 0.5
        W[:, 3] = -centers[:, 1] + 0.5
        W[:, 4] = centers[:, 2] + 0.5
        W[:, 5] = -centers[:, 2] + 0.5
        
        # The soft-min was applied to `wall_dists` which is `min(W, axis=1)`.
        # This is problematic. The soft-min should be applied to the *raw* distances to be smooth everywhere.
        # If I apply soft-min to `min(W)`, it's non-differentiable at points where two walls are equally close.
        
        # Better approach: Apply soft-min to the *entire* set of constraints:
        # All pair distances / 2 AND all 6*N wall distances.
        
        # Let's refactor the objective to use raw wall distances.
        
        # Re-do logic inside objective:
        # 1. Compute pair dists / 2.
        # 2. Compute 6*N wall dists.
        # 3. Concatenate into V.
        # 4. Soft-min of V.
        # 5. Gradient back-propagates to pairs and walls.
        
        # This is more expensive but smoother.
        # N=100, pairs ~5000, walls 600. Total ~5600 values.
        # Soft-min on 5600 values is fine.
        
        # Let's rewrite the inner logic of objective_and_grad to be correct.
        
        # We already calculated `obj` and `grad_flat` for pairs above, but `grad_wall` was wrong.
        # Let's discard the previous calculation and do it correctly.
        
        # --- Corrected Gradient Calculation ---
        
        # Pair part is correct in logic, but let's ensure we don't double count or miss.
        
        # Wall part:
        # V_walls = W (N x 6)
        # Flatten W to V_walls_flat
        V_walls_flat = W.flatten()
        
        # We need the gradient of the soft-min of the *combined* vector V = [V_pairs, V_walls_flat].
        # Let's recompute soft-min on the combined vector.
        
        V_pairs = pair_dists
        V_all = np.concatenate([V_pairs, V_walls_flat])
        
        val, grad_all = _softmin_grad(V_all, beta)
        obj = -val
        
        grad_flat = np.zeros_like(x_flat)
        
        # 1. Pair contributions
        grad_pair = grad_all[:len(V_pairs)]
        np.add.at(grad_flat, i_idx, grad_pair[:, None] * 0.5 * unit_vecs)
        np.add.at(grad_flat, j_idx, -grad_pair[:, None] * 0.5 * unit_vecs)
        
        # 2. Wall contributions
        grad_wall_flat = grad_all[len(V_pairs):]
        grad_wall = grad_wall_flat.reshape(N, 6)
        
        # grad_wall[i, k] is the weight of wall k for sphere i.
        # We need to map wall index k to coordinate axis and sign.
        # k=0: x+0.5 -> grad_x = 1
        # k=1: -x+0.5 -> grad_x = -1
        # k=2: y+0.5 -> grad_y = 1
        # k=3: -y+0.5 -> grad_y = -1
        # k=4: z+0.5 -> grad_z = 1
        # k=5: -z+0.5 -> grad_z = -1
        
        # grad_flat[:, 0] += grad_wall[:, 0] * 1.0 - grad_wall[:, 1] * 1.0
        # grad_flat[:, 1] += grad_wall[:, 2] * 1.0 - grad_wall[:, 3] * 1.0
        # grad_flat[:, 2] += grad_wall[:, 4] * 1.0 - grad_wall[:, 5] * 1.0
        
        grad_flat[:, 0] += grad_wall[:, 0] - grad_wall[:, 1]
        grad_flat[:, 1] += grad_wall[:, 2] - grad_wall[:, 3]
        grad_flat[:, 2] += grad_wall[:, 4] - grad_wall[:, 5]
        
        return obj, grad_flat

    # Optimization Loop with adaptive beta
    # Start with low beta (smooth), increase beta (sharp)
    betas = [50, 100, 200, 500, 1000]
    current_centers = best_centers.copy()
    
    for beta in betas:
        if time.time() > deadline - 20.0:
            break
            
        # Flatten current centers
        x0 = current_centers.flatten()
        
        # Bounds
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            res = minimize(
                lambda x: objective_and_grad(x, beta)[0],
                x0,
                jac=lambda x: objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < 0: # Check if it made sense
                new_centers = res.x.reshape(N, 3)
                new_radius = _compute_radius(new_centers)
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_centers = new_centers.copy()
                    current_centers = new_centers.copy()
        except Exception:
            pass

    # Final polish with very high beta if time permits
    if time.time() < deadline - 10.0:
        beta = 2000
        x0 = current_centers.flatten()
        bounds = [(-0.5, 0.5)] * (N * 3)
        try:
            res = minimize(
                lambda x: objective_and_grad(x, beta)[0],
                x0,
                jac=lambda x: objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-10}
            )
            new_centers = res.x.reshape(N, 3)
            new_radius = _compute_radius(new_centers)
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = new_centers.copy()
        except Exception:
            pass

    return best_centers
# EVOLVE-BLOCK-END
