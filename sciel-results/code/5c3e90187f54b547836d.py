import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin for 120s limit

    # 1. Initialization
    # Use a 5x5x5 grid, pick 100 points
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    # Generate all 125 points
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # We need 100 points.
    # A good heuristic: pick points that are not on the very edge?
    # Or just random selection with a fixed seed for reproducibility.
    # Let's try to pick the 100 points that form the "best" initial packing.
    # For simplicity, we'll take the first 100, or remove the 25 corners?
    # Corners: (0,0,0) type. In 5x5x5, corners are (0,0,0), (0,0,4)...
    # Let's just shuffle and take 100, or take all except the 25 with smallest "internal" volume?
    # Actually, a uniform distribution is fine. The optimizer will fix it.
    # Let's remove the 25 points closest to the corners to give them more room?
    # No, corners are good for packing if the sphere is small.
    # Let's just take a random subset of 100 from the 125.
    np.random.seed(42)
    indices = np.random.choice(125, 100, replace=False)
    centers = grid[indices].copy()
    
    # Add small noise to break symmetry
    centers += np.random.normal(0, 0.01, size=(N, 3))
    
    # Clip to cube
    centers = np.clip(centers, -0.5, 0.5)

    # 2. Optimization
    beta = 10.0
    step_size = 0.005
    max_iters = 5000
    
    for i in range(max_iters):
        if time.time() - start_time > timeout:
            break
            
        # Compute clearances
        # Pairwise distances
        # c: (N, 3)
        # diff: (N, N, 3)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Avoid division by zero
        np.fill_diagonal(dists, 1.0)
        
        # Pairwise clearances: 0.5 * dist
        pair_clearances = 0.5 * dists
        # Exclude self (diagonal)
        # We will sum over i < j
        
        # Boundary clearances
        # c_ik in [-0.5, 0.5]
        # clearance = 0.5 - |c_ik|
        bound_clearances = 0.5 - np.abs(centers) # (N, 3)
        
        # Combine all clearances into one vector for soft-min
        # Number of pairs: N*(N-1)/2 = 4950
        # Number of boundary terms: N*3 = 300
        # Total ~ 5250 terms
        
        # For efficiency, we can compute the log-sum-exp manually
        # S = -1/beta * log( sum(exp(-beta * x)) )
        
        # Get upper triangle indices for pairs
        i_idx, j_idx = np.triu_indices(N, k=1)
        pair_vals = pair_clearances[i_idx, j_idx]
        
        # Flatten boundary values
        bound_vals = bound_clearances.flatten()
        
        # Combine
        all_vals = np.concatenate([pair_vals, bound_vals])
        
        # Compute weights w_k = exp(-beta * x_k) / sum(exp(-beta * x))
        # Numerical stability: subtract max
        max_val = np.max(all_vals)
        # Wait, soft-min of x is -1/beta log sum exp(-beta x).
        # If x is large, exp(-beta x) is small.
        # We want to maximize the minimum x.
        # The gradient of the soft-min w.r.t x_k is w_k.
        # w_k = exp(-beta x_k) / Z.
        
        # To avoid underflow, we can shift by max?
        # Actually, if x is large, exp(-beta x) is 0.
        # The "active" constraints are the ones with small x (small clearance).
        # So we should shift by MIN?
        # Let's check:
        # min(x) = -1/beta log sum exp(-beta x)
        # If x = [1, 100], beta=10.
        # exp(-10) = 4.5e-5, exp(-1000) = 0.
        # sum = 4.5e-5.
        # -1/10 log(4.5e-5) = -1/10 * (-10.0) = 1.0. Correct.
        
        # So we don't need to shift by max. We need to be careful with underflow.
        # But N is small.
        
        exp_vals = np.exp(-beta * all_vals)
        Z = np.sum(exp_vals)
        weights = exp_vals / Z
        
        # Gradient of S w.r.t centers
        # dS/dc = sum_k w_k * (dx_k/dc)
        
        grad = np.zeros_like(centers)
        
        # 1. Boundary terms
        # bound_vals corresponds to centers.flatten()
        # dx/dc = -sign(c)
        bound_weights = weights[len(pair_vals):] # (N, 3)
        bound_weights = bound_weights.reshape(N, 3)
        
        # grad += sum over k of w_ik * (-sign(c_ik))
        grad -= bound_weights * np.sign(centers)
        
        # 2. Pairwise terms
        # pair_vals corresponds to dists[i,j]
        # dx_ij/dc_i = 0.5 * (c_i - c_j) / ||c_i - c_j||
        # dx_ij/dc_j = 0.5 * (c_j - c_i) / ||c_i - c_j||
        
        pair_weights = weights[:len(pair_vals)] # (N_pairs,)
        
        # We need to accumulate gradients for i and j
        # diff[i_idx, j_idx] is c_i - c_j
        # unit_vec = diff / dist
        
        # dists[i_idx, j_idx] is the distance
        p_dists = dists[i_idx, j_idx]
        
        # Avoid div by 0 (should not happen if init is good, but safe)
        p_dists = np.where(p_dists < 1e-9, 1e-9, p_dists)
        
        # diff vectors
        p_diffs = diff[i_idx, j_idx] # (N_pairs, 3)
        
        # Unit vectors
        unit_vecs = p_diffs / p_dists[:, np.newaxis]
        
        # Gradient contribution for c_i: w * 0.5 * unit_vec
        # Gradient contribution for c_j: w * 0.5 * (-unit_vec)
        
        # We use np.add.at for accumulation
        np.add.at(grad, i_idx, 0.5 * pair_weights[:, np.newaxis] * unit_vecs)
        np.add.at(grad, j_idx, -0.5 * pair_weights[:, np.newaxis] * unit_vecs)
        
        # Update centers
        centers += step_size * grad
        
        # Clip to cube to prevent divergence
        centers = np.clip(centers, -0.5, 0.5)
        
        # Annealing
        if i % 50 == 0:
            beta *= 1.2
            step_size *= 0.95
            
        # Stop if beta is too high
        if beta > 1000:
            break
            
    return centers

# EVOLVE-BLOCK-END
