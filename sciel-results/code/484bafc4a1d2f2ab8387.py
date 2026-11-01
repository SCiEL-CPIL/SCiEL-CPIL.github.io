import numpy as np
import time

N = 200

def construct():
    start_time = time.time()
    time_limit = 145.0  # 5s safety margin for 150s limit
    
    # Initialize with a grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N].copy()
    # Add small random noise to break symmetry
    np.random.seed(42) # For reproducibility in evaluation? No, evaluator might run multiple times. 
    # But consistent behavior is good.
    centers += 0.02 * np.random.randn(N, 2)
    centers = np.clip(centers, -0.49, 0.49)

    # Precompute pair indices
    i_idx, j_idx = np.triu_indices(N, k=1)
    pairs_i = i_idx
    pairs_j = j_idx
    
    # Optimization parameters
    beta = 10.0
    beta_final = 100.0
    lr = 0.005
    lr_decay = 0.9995
    
    while time.time() - start_time < time_limit:
        # Compute pairwise differences
        diffs = centers[pairs_i] - centers[pairs_j]
        dists_sq = np.sum(diffs**2, axis=1)
        dists = np.sqrt(dists_sq + 1e-12)
        
        # Boundary distances
        x = centers[:, 0]
        y = centers[:, 1]
        
        d_xp = 0.5 - x
        d_xm = 0.5 + x
        d_yp = 0.5 - y
        d_ym = 0.5 + y
        
        d_xp = np.maximum(d_xp, 1e-12)
        d_xm = np.maximum(d_xm, 1e-12)
        d_yp = np.maximum(d_yp, 1e-12)
        d_ym = np.maximum(d_ym, 1e-12)

        # Exponentials
        exp_pairs = np.exp(-beta * dists)
        exp_bx_p = np.exp(-beta * d_xp)
        exp_bx_m = np.exp(-beta * d_xm)
        exp_by_p = np.exp(-beta * d_yp)
        exp_by_m = np.exp(-beta * d_ym)
        
        Z = np.sum(exp_pairs) + np.sum(exp_bx_p) + np.sum(exp_bx_m) + np.sum(exp_by_p) + np.sum(exp_by_m)
        
        if Z == 0:
            break
            
        grads = np.zeros_like(centers)
        
        # Pair gradients
        w_pairs = exp_pairs / Z
        vec_pairs = diffs / dists[:, np.newaxis]
        
        np.add.at(grads, pairs_i, w_pairs[:, np.newaxis] * vec_pairs)
        np.add.at(grads, pairs_j, -w_pairs[:, np.newaxis] * vec_pairs)
        
        # Boundary gradients
        w_bxp = exp_bx_p / Z
        w_bxm = exp_bx_m / Z
        w_byp = exp_by_p / Z
        w_by_m = exp_by_m / Z
        
        grads[:, 0] += (-w_bxp + w_bxm)
        grads[:, 1] += (-w_byp + w_by_m)
        
        # Update
        centers += lr * grads
        centers = np.clip(centers, -0.5, 0.5)
        
        # Decay
        lr *= lr_decay
        beta = min(beta * 1.001, beta_final)
        
    return centers
