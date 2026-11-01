def _soft_min_pgd(centers, beta, steps, time_budget, rng):
    """
    Maximize the soft-min of pairwise and wall distances.
    """
    start = time.time()
    cur = centers.copy()
    best = cur.copy()
    best_score = -np.inf
    
    step = 0.005
    damping = 0.7
    momentum = np.zeros_like(cur)
    
    # Precompute indices for pairs
    iu, ju = np.triu_indices(N, k=1)
    
    for it in range(steps):
        if time.time() - start > time_budget:
            break
        
        # Compute pairwise distances
        diff = cur[iu] - cur[ju]
        dists = np.linalg.norm(diff, axis=1)
        
        # Compute wall distances
        # dist to left wall: x + 0.5
        # dist to right wall: 0.5 - x
        # dist to bottom wall: y + 0.5
        # dist to top wall: 0.5 - y
        wall_dists = np.concatenate([
            cur[:, 0] + 0.5,
            0.5 - cur[:, 0],
            cur[:, 1] + 0.5,
            0.5 - cur[:, 1]
        ])
        
        # All distances
        all_dists = np.concatenate([dists, wall_dists])
        
        # Soft min: -1/beta * log(sum(exp(-beta * d)))
        # To avoid overflow, subtract max
        max_d = np.max(all_dists)
        exp_terms = np.exp(-beta * (all_dists - max_d))
        log_sum = np.log(np.sum(exp_terms))
        soft_min_val = max_d - (1.0 / beta) * log_sum
        
        if soft_min_val > best_score:
            best_score = soft_min_val
            best = cur.copy()
            
        # Gradient of soft min w.r.t. centers
        # d(soft_min)/d(d_k) = exp(-beta(d_k - max_d)) / sum(exp(...))
        # Note: gradient of -log(sum(exp(-beta d))) is -1/sum * sum(exp(-beta d) * (-beta) * dd/dx)
        # Let S = sum(exp(-beta(d_k - max_d)))
        # d(soft_min)/dx = -1/beta * (1/S) * d(S)/dx
        # d(S)/dx = sum( exp(-beta(d_k - max_d)) * (-beta) * d(d_k)/dx )
        # So d(soft_min)/dx = (1/S) * sum( exp(-beta(d_k - max_d)) * d(d_k)/dx )
        
        weights = exp_terms / np.sum(exp_terms)
        
        # Gradient of distances w.r.t. centers
        grad = np.zeros_like(cur)
        
        # Pairwise gradients
        # d(d_ij)/d(c_i) = (c_i - c_j) / d_ij
        # d(d_ij)/d(c_j) = (c_j - c_i) / d_ij
        unit_vecs = diff / dists[:, np.newaxis]
        
        # Accumulate gradients for pairs
        # For each pair (i, j), weight w:
        # grad[i] += w * unit_vec
        # grad[j] -= w * unit_vec
        # We can use np.add.at for this
        np.add.at(grad, iu, weights[:, np.newaxis] * unit_vecs)
        np.add.at(grad, ju, -weights[:, np.newaxis] * unit_vecs)
        
        # Wall gradients
        # d(x+0.5)/dx = 1
        # d(0.5-x)/dx = -1
        # d(y+0.5)/dy = 1
        # d(0.5-y)/dy = -1
        
        # weights are ordered: [pair_dists..., wall_dists...]
        # wall_dists order: [x_lo..., x_hi..., y_lo..., y_hi...]
        w_wall = weights[len(dists):]
        w_x_lo = w_wall[:N]
        w_x_hi = w_wall[N:2*N]
        w_y_lo = w_wall[2*N:3*N]
        w_y_hi = w_wall[3*N:]
        
        grad[:, 0] += (w_x_lo - w_x_hi)[:, np.newaxis] * np.ones((1, 1)) # Wait, w_x_lo is vector of N, need to add to each row
        # Correctly:
        grad[:, 0] += (w_x_lo - w_x_hi)
        grad[:, 1] += (w_y_lo - w_y_hi)
        
        # Update
        momentum = damping * momentum + grad
        cur += step * momentum
        
        # Project to box (strictly inside to avoid boundary issues)
        cur = np.clip(cur, -0.4999, 0.4999)
        
        # Adaptive step
        # Optional: check if soft_min increased
        # For simplicity, keep fixed step or simple decay
        if it % 100 == 0:
            step *= 0.99

    return best
