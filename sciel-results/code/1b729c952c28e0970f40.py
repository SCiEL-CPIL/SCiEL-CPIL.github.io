import numpy as np
import time

N = 200

def construct():
    start_time = time.time()
    deadline = 145.0 # 145s to be safe
    
    # Initialization: Grid with noise
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Add small noise to break symmetry
    np.random.seed(42)
    centers += np.random.normal(0, 0.001, centers.shape)
    
    # Precompute indices for pairs
    # We can vectorize the distance calculation
    # But for gradient accumulation, it's easier to loop or use sparse matrices
    # Given N=200, a loop over pairs might be slow in Python.
    # Let's try to vectorize the gradient accumulation as much as possible.
    
    # Pairs (i, j) with i < j
    i_idx, j_idx = np.triu_indices(N, k=1)
    n_pairs = len(i_idx)
    
    # Boundary indices
    # For each point i, distance to 4 boundaries
    # d_ix = 0.5 - abs(c_ix)
    # d_iy = 0.5 - abs(c_iy)
    
    t = 10.0 # Initial temperature
    t_min = 1000.0 # Final temperature (sharper min)
    steps = 2000
    
    # Learning rate
    alpha = 0.01
    
    best_radius = -1
    best_centers = centers.copy()
    
    # We need to compute soft-min and gradient
    # S = -1/t * log(sum(exp(-t * d_k)))
    # grad_S_i = sum_k w_k * grad_d_k_i
    # where w_k = exp(-t * d_k) / sum(exp(-t * d_l))
    
    # To avoid underflow/overflow with exp, we can subtract max
    # Let M = max(-t * d_k) = -t * min(d_k)
    # sum(exp(-t * d_k)) = exp(M) * sum(exp(-t * d_k - M))
    # log(sum(...)) = M + log(sum(exp(...)))
    # S = -1/t * (M + log(sum(exp(...))))
    
    for step in range(steps):
        if time.time() > deadline:
            break
            
        # 1. Compute pairwise distances
        diff = centers[i_idx] - centers[j_idx]
        dists = np.linalg.norm(diff, axis=1)
        
        # 2. Compute boundary distances
        # x-boundaries
        d_x_pos = 0.5 - centers[:, 0] # dist to x=0.5
        d_x_neg = 0.5 + centers[:, 0] # dist to x=-0.5
        d_y_pos = 0.5 - centers[:, 1] # dist to y=0.5
        d_y_neg = 0.5 + centers[:, 1] # dist to y=-0.5
        
        # Combine all distances into one array for soft-min
        # d_all = [dists, d_x_pos, d_x_neg, d_y_pos, d_y_neg]
        # This is 20000 + 800 = 20800 values
        
        d_all = np.concatenate([dists, d_x_pos, d_x_neg, d_y_pos, d_y_neg])
        
        # 3. Compute soft-min and weights
        # Handle zero distances
        d_safe = np.maximum(d_all, 1e-12)
        
        # Soft min calculation
        # We want to maximize min(d).
        # f = -1/t * log(sum(exp(-t * d)))
        
        # For numerical stability
        # Let u_k = -t * d_k
        # max_u = max(u)
        # sum_exp = sum(exp(u - max_u))
        # log_sum_exp = max_u + log(sum_exp)
        # f = -1/t * log_sum_exp
        
        u = -t * d_safe
        max_u = np.max(u)
        
        # If max_u is very large negative, exp(u - max_u) is fine
        # If max_u is very large positive, exp(u - max_u) is fine
        exp_u = np.exp(u - max_u)
        sum_exp = np.sum(exp_u)
        log_sum_exp = max_u + np.log(sum_exp)
        
        soft_min_val = -log_sum_exp / t
        
        # Weights w_k = exp(u_k) / sum(exp(u_l)) = exp(u_k - max_u) / sum_exp
        w_all = exp_u / sum_exp
        
        # 4. Compute gradients
        # grad_centers = np.zeros((N, 2))
        
        # Pairwise gradients
        # For pair (i, j) with distance d_ij and weight w_ij:
        # grad_i += w_ij * (c_i - c_j) / d_ij
        # grad_j += w_ij * (c_j - c_i) / d_ij
        
        # We have w_pairs = w_all[:n_pairs]
        w_pairs = w_all[:n_pairs]
        
        # Unit vectors
        # diff = c_i - c_j
        # u_vec = diff / dists
        # Avoid division by zero
        dists_safe = np.maximum(dists, 1e-12)
        u_vec = diff / dists_safe[:, np.newaxis]
        
        # Weighted unit vectors
        wu_vec = w_pairs[:, np.newaxis] * u_vec
        
        # Accumulate gradients
        # np.add.at(grad_centers, i_idx, wu_vec)
        # np.add.at(grad_centers, j_idx, -wu_vec)
        
        grad_centers = np.zeros((N, 2))
        np.add.at(grad_centers, i_idx, wu_vec)
        np.add.at(grad_centers, j_idx, -wu_vec)
        
        # Boundary gradients
        # d_x_pos = 0.5 - c_x. grad wrt c_x is -1.
        # d_x_neg = 0.5 + c_x. grad wrt c_x is 1.
        # d_y_pos = 0.5 - c_y. grad wrt c_y is -1.
        # d_y_neg = 0.5 + c_y. grad wrt c_y is 1.
        
        w_x_pos = w_all[n_pairs : n_pairs + N]
        w_x_neg = w_all[n_pairs + N : n_pairs + 2*N]
        w_y_pos = w_all[n_pairs + 2*N : n_pairs + 3*N]
        w_y_neg = w_all[n_pairs + 3*N : n_pairs + 4*N]
        
        # grad_x from x-boundaries
        grad_centers[:, 0] += -w_x_pos + w_x_neg
        # grad_y from y-boundaries
        grad_centers[:, 1] += -w_y_pos + w_y_neg
        
        # 5. Update centers
        # Simple gradient ascent
        # Step size might need to be adaptive
        # alpha = 0.01 * (1.0 / (1.0 + step/100.0)) # Decay?
        
        centers += alpha * grad_centers
        
        # Clamp centers to stay inside square (optional, helps stability)
        centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
        centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
        
        # Anneal temperature
        # Exponential decay
        t *= 1.01
        if t > t_min:
            t = t_min
            
        # Track best
        # The actual radius is min(d_all)
        current_radius = np.min(d_all)
        if current_radius > best_radius:
            best_radius = current_radius
            best_centers = centers.copy()
            
    return best_centers
