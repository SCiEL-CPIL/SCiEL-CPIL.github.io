"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _score(centers):
    """Calculate the score (radius) for a given set of centers."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    
    # Compute pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Min pairwise distance
    min_pair_dist = np.min(dists[np.triu_indices(N, k=1)])
    
    # Min distance to walls
    wall_dists = 0.5 - np.abs(centers)
    min_wall_dist = np.min(wall_dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _smooth_radius(centers, beta=20.0):
    """
    Compute a smooth approximation of the minimum radius.
    r_smooth = -1/beta * log( sum(exp(-beta * d_ij/2)) + sum(exp(-beta * d_wall)) )
    This is large when min distance is large.
    """
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    
    # Pairwise terms: exp(-beta * d_ij / 2)
    # d_ij = sqrt(dists_sq)
    # We want to avoid sqrt if possible for speed, but exp(-beta * sqrt(x)/2) is fine.
    # Actually, let's use the squared distance in the exponent? No, the radius is linear.
    # Let's stick to distances.
    dists = np.sqrt(np.maximum(dists_sq, 1e-12))
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    min_wall_per_sphere = np.min(wall_dists, axis=1)
    
    # Combine all constraints into a single sum of exponentials
    # We want to maximize min(d_ij/2, d_wall).
    # Let f(x) = min(x). log-sum-exp approximates max.
    # So we compute log-sum-exp of -beta * x_k.
    # x_k are d_ij/2 and d_wall.
    
    pair_terms = np.exp(-beta * dists[mask] / 2.0)
    wall_terms = np.exp(-beta * min_wall_per_sphere)
    
    total_sum = np.sum(pair_terms) + np.sum(wall_terms)
    
    if total_sum == 0:
        return 0.0
        
    r_smooth = -1.0 / beta * np.log(total_sum)
    return r_smooth

def _gradient_smooth_radius(centers, beta=20.0):
    """
    Compute gradient of the smooth radius with respect to centers.
    We want to maximize r_smooth, so we will add gradient * step_size.
    """
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    dists = np.sqrt(np.maximum(dists_sq, 1e-12))
    
    mask = ~np.eye(N, dtype=bool)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    min_wall_idx = np.argmin(wall_dists, axis=1)
    min_wall_per_sphere = wall_dists[np.arange(N), min_wall_idx]
    
    # Weights for each constraint
    pair_terms = np.exp(-beta * dists[mask] / 2.0)
    wall_terms = np.exp(-beta * min_wall_per_sphere)
    
    total_sum = np.sum(pair_terms) + np.sum(wall_terms)
    
    if total_sum == 0:
        return np.zeros_like(centers)
    
    # Gradient of log-sum-exp
    # d/dx log(sum exp(f_k)) = sum(exp(f_k) * f'_k) / sum(exp(f_k))
    
    grad = np.zeros_like(centers)
    
    # Pairwise gradients
    # f_ij = -beta * d_ij / 2
    # f'_ij = -beta/2 * d/dx_i (d_ij)
    # d/dx_i (d_ij) = (x_i - x_j) / d_ij
    # So f'_ij = -beta/2 * (x_i - x_j) / d_ij
    
    # We need to sum over all j for each i.
    # Let's compute the force vector for each i.
    
    # pair_terms is flattened. We need to map back to i, j.
    # It's easier to compute the gradient directly in a loop or vectorized way over i, j.
    
    # Vectorized approach:
    # For each i, sum over j != i of weight_ij * grad_ij
    
    # weight_ij = pair_terms_flat / total_sum
    # But pair_terms is masked.
    
    # Let's compute weights matrix W of shape (N, N)
    W = np.zeros((N, N))
    W[mask] = pair_terms / total_sum
    
    # Gradient contribution from pairs:
    # grad_i += sum_j W_ij * (-beta/2) * (x_i - x_j) / d_ij
    
    # Note: d/dx_i (d_ij) = (x_i - x_j)/d_ij
    # So term is -beta/2 * (x_i - x_j)/d_ij
    
    # Let's compute (x_i - x_j)/d_ij for all i, j
    # diff is x_i - x_j
    # dists is d_ij
    
    # Avoid division by zero (diagonal is 0, but W is 0 there)
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vecs = diff / dists[:, :, np.newaxis]
    
    # unit_vecs[i, j] = (x_i - x_j) / d_ij
    # We want sum_j W_ij * (-beta/2) * unit_vecs[i, j]
    
    # Sum over axis 1 (j)
    pair_grad = np.sum(W[:, :, np.newaxis] * unit_vecs, axis=1) * (-beta / 2.0)
    
    # Wall gradients
    # f_wall_i = -beta * d_wall_i
    # d_wall_i = 0.5 - |x_ik| for the min axis k
    # d/dx_ik (0.5 - |x_ik|) = -sign(x_ik)
    # So f'_wall_i = -beta * (-sign(x_ik)) = beta * sign(x_ik)
    
    wall_weights = wall_terms / total_sum
    
    # grad_i += wall_weights[i] * beta * sign(x_i) for the specific axis k that was min
    # Actually, the min wall distance is min_k (0.5 - |x_ik|).
    # The gradient is non-zero only for the axis k that achieves the minimum.
    # If multiple axes achieve the min, we sum the gradients?
    # The function is min_k g_k(x). The subgradient is the convex hull of gradients of active constraints.
    # For smooth approximation, we use the weighted sum of exponentials.
    # The "active" constraint is effectively smoothed.
    # However, `min_wall_per_sphere` was computed as the min.
    # To be consistent with the smooth function defined as sum of exps of ALL wall distances?
    # My `_smooth_radius` used `min_wall_per_sphere`. This is not differentiable if we take min.
    # Let's refine `_smooth_radius` to use sum of exps of ALL 3 wall distances per sphere?
    # No, the constraint is `min_k (0.5 - |x_ik|)`.
    # If I use `min`, the gradient is tricky.
    # Better smooth function:
    # r = min( min_{i<j} d_ij/2, min_i min_k d_ik )
    # Let's use log-sum-exp over ALL constraints:
    # Constraints: d_ij/2 for i<j, and d_ik for all i,k.
    
    # Let's redefine `_smooth_radius` and `_gradient` to use all constraints explicitly.
    # This is cleaner.
    
    return pair_grad # Placeholder, will rewrite function

def _smooth_obj_and_grad(centers, beta=20.0):
    """
    Computes smooth objective and its gradient.
    Objective: - log( sum exp(-beta * c_k) )
    where c_k are all constraints: d_ij/2 for pairs, d_ik for walls.
    Maximizing this is equivalent to maximizing min(c_k).
    """
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    dists = np.sqrt(np.maximum(dists_sq, 1e-12))
    
    mask = ~np.eye(N, dtype=bool)
    
    # Constraints
    # Pairs: d_ij / 2
    pair_constraints = dists[mask] / 2.0
    
    # Walls: 0.5 - |x_ik|
    wall_constraints = 0.5 - np.abs(centers)
    
    # Flatten constraints
    all_constraints = np.concatenate([pair_constraints, wall_constraints.flatten()])
    
    # Objective: -1/beta * log(sum(exp(-beta * c)))
    # To avoid underflow, subtract max
    max_c = np.max(all_constraints)
    exp_terms = np.exp(-beta * (all_constraints - max_c))
    
    sum_exp = np.sum(exp_terms)
    
    if sum_exp == 0:
        return 0.0, np.zeros_like(centers)
        
    obj_val = -1.0 / beta * (np.log(sum_exp) - beta * max_c)
    
    # Gradient
    # d/dx obj = -1/beta * ( sum(exp(-beta(c_k - max_c)) * d/dx(-beta(c_k - max_c))) / sum_exp )
    # Note: max_c is treated as constant in derivative? No, max_c depends on x.
    # However, for large beta, the term with max_c dominates.
    # A safer gradient for log-sum-exp is:
    # grad = sum( w_k * grad(c_k) ) where w_k = exp(-beta c_k) / sum(exp(-beta c_k))
    # Wait, obj = -1/beta log(sum exp(-beta c_k)).
    # d obj/dx = -1/beta * ( 1/S * sum( exp(-beta c_k) * (-beta) * d c_k/dx ) )
    # d obj/dx = sum( exp(-beta c_k) * d c_k/dx ) / S
    # Let w_k = exp(-beta c_k) / S.
    # grad = sum( w_k * d c_k/dx )
    
    # We need to compute w_k for all constraints.
    # We already have exp_terms for (c_k - max_c).
    # w_k = exp(-beta(c_k - max_c)) / sum_exp
    
    weights = exp_terms / sum_exp
    
    # Split weights back
    n_pairs = len(pair_constraints)
    w_pairs = weights[:n_pairs]
    w_walls = weights[n_pairs:]
    
    grad = np.zeros_like(centers)
    
    # Pair gradients
    # c_ij = d_ij / 2
    # d c_ij/d x_i = 1/2 * (x_i - x_j) / d_ij
    
    # We need to map w_pairs back to i, j.
    # w_pairs is flattened from mask.
    # Let's compute the sum over j for each i.
    
    # Create a weight matrix W of shape (N, N)
    W = np.zeros((N, N))
    W[mask] = w_pairs
    
    # unit_vecs[i, j] = (x_i - x_j) / d_ij
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vecs = diff / dists[:, :, np.newaxis]
        
    # d c_ij/d x_i = 0.5 * unit_vecs[i, j]
    # grad_i += sum_j W_ij * 0.5 * unit_vecs[i, j]
    
    pair_grad = np.sum(W[:, :, np.newaxis] * unit_vecs, axis=1) * 0.5
    
    # Wall gradients
    # c_ik = 0.5 - |x_ik|
    # d c_ik/d x_ik = -sign(x_ik)
    
    # w_walls is flattened from wall_constraints (N, 3)
    W_walls = w_walls.reshape(N, 3)
    
    # d c_ik/d x_ik = -sign(x_ik)
    signs = np.sign(centers)
    
    # grad_i += sum_k W_walls[i, k] * (-signs[i, k])
    wall_grad = -np.sum(W_walls * signs, axis=1)
    
    grad = pair_grad + wall_grad
    
    return obj_val, grad

def _local_search(centers, time_limit=5.0):
    """
    Direct local search maximizing the minimum distance.
    """
    start_time = time.time()
    centers = centers.copy()
    current_score = _score(centers)
    best_centers = centers.copy()
    best_score = current_score
    
    iu = np.triu_indices(N, k=1)
    step_size = 0.02
    
    for iteration in range(5000):
        if time.time() - start_time > time_limit:
            break
            
        # Compute min pair and min wall
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        pair_dists_sq = dists_sq[iu]
        min_pair_dist_sq = np.min(pair_dists_sq)
        min_pair_dist = np.sqrt(min_pair_dist_sq)
        
        wall_dists = 0.5 - np.abs(centers)
        min_wall_dist_per_sphere = np.min(wall_dists, axis=1)
        global_min_wall_dist = np.min(min_wall_dist_per_sphere)
        
        current_r = min(min_pair_dist / 2.0, global_min_wall_dist)
        
        # Find critical indices
        min_pair_idx = np.argmin(pair_dists_sq)
        i, j = iu[0][min_pair_idx], iu[1][min_pair_idx]
        min_wall_idx = np.argmin(min_wall_dist_per_sphere)
        
        candidates = [i, j, min_wall_idx]
        # Add random sphere
        if np.random.rand() < 0.1:
            candidates.append(np.random.randint(N))
        
        improved = False
        for idx in candidates:
            # Try to move idx
            for _ in range(10): # 10 random trials per candidate
                direction = np.random.normal(0, 1, 3)
                direction /= np.linalg.norm(direction)
                new_pos = centers[idx] + direction * step_size
                
                if np.any(new_pos < -0.5) or np.any(new_pos > 0.5):
                    continue
                    
                old_pos = centers[idx]
                centers[idx] = new_pos
                
                new_score = _score(centers)
                
                if new_score > current_r:
                    current_r = new_score
                    if current_r > best_score:
                        best_score = current_r
                        best_centers = centers.copy()
                    improved = True
                    break # Accept this move, try next candidate
                else:
                    centers[idx] = old_pos
            
            if improved:
                step_size *= 0.9 # Refine
            else:
                step_size *= 1.05 # Explore
                step_size = min(step_size, 0.1)
                
        # Reset step size occasionally
        if iteration % 50 == 0:
             step_size = 0.02

    return best_centers, best_score

def _optimize_smooth(centers, time_limit=10.0, beta=20.0):
    """
    Optimize using gradient ascent on the smooth objective.
    """
    start_time = time.time()
    centers = centers.copy()
    lr = 0.01
    momentum = 0.9
    velocity = np.zeros_like(centers)
    best_obj, _ = _smooth_obj_and_grad(centers, beta)
    best_centers = centers.copy()
    
    while time.time() - start_time < time_limit:
        obj, grad = _smooth_obj_and_grad(centers, beta)
        
        if obj > best_obj:
            best_obj = obj
            best_centers = centers.copy()
            
        # Update velocity
        velocity = momentum * velocity + lr * grad
        
        # Update centers
        centers += velocity
        
        # Clip to cube
        centers = np.clip(centers, -0.5, 0.5)
        
        # Decay learning rate slightly
        lr *= 0.999
        
    return best_centers

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_score = 0.0
    best_centers = None
    
    rng = np.random.default_rng(42)
    
    # Strategy: Multiple restarts with different initializations
    # 1. Jittered Grid
    # 2. Random
    # 3. FCC-like (if possible to generate easily, otherwise just grid)
    
    # Grid Initialization
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # We have 64 points in 4x4x4 grid. We need 50.
    # Remove 14 random points.
    
    for restart in range(10):
        if time.time() > deadline:
            break
            
        if restart < 5:
            # Grid Pruning
            idx = rng.permutation(len(grid))[:N]
            centers = grid[idx].copy()
            centers += rng.normal(0, 0.02, centers.shape)
        else:
            # Random
            centers = rng.uniform(-0.45, 0.45, (N, 3))
            
        centers = np.clip(centers, -0.49, 0.49)
        
        # 1. Smooth Optimization
        time_left = deadline - time.time()
        if time_left > 20:
            opt_time = min(10.0, time_left / 3.0)
            centers = _optimize_smooth(centers, time_limit=opt_time)
            centers = np.clip(centers, -0.49, 0.49)
        
        # 2. Local Search
        time_left = deadline - time.time()
        if time_left > 10:
            ls_time = min(5.0, time_left / 2.0)
            refined_centers, score = _local_search(centers, time_limit=ls_time)
            
            if score > best_score:
                best_score = score
                best_centers = refined_centers.copy()
            
    # Final Polish on the best candidate
    if best_centers is not None and (deadline - time.time()) > 10:
        final_time = deadline - time.time() - 5
        refined_centers, score = _local_search(best_centers, time_limit=final_time)
        if score > best_score:
            best_score = score
            best_centers = refined_centers.copy()

    if best_centers is None:
        # Fallback
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers
# EVOLVE-BLOCK-END
