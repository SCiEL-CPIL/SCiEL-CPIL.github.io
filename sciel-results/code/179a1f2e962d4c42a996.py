def _energy_and_grad(positions, beta=1.0):
    positions = np.asarray(positions).reshape(-1, 3)
    n = positions.shape[0]
    
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    wall_dists = np.stack([
        0.5 - x, 0.5 + x,
        0.5 - y, 0.5 + y,
        0.5 - z, 0.5 + z
    ], axis=1)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    diffs = positions[i_idx] - positions[j_idx]
    pair_dists = np.linalg.norm(diffs, axis=1)
    
    all_dists = np.concatenate([wall_dists.flatten(), pair_dists])
    
    max_d = np.max(all_dists)
    exp_terms = np.exp(-beta * (all_dists - max_d))
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp
    
    # Objective to minimize: -S
    obj = max_d + (1.0 / beta) * np.log(sum_exp)
    
    # Gradient of S: sum(w * grad_d)
    # Gradient of -S: -sum(w * grad_d)
    
    grad_s = np.zeros_like(positions)
    
    w_wall = weights[:n*6].reshape(n, 6)
    grad_s[:, 0] += -w_wall[:, 0] + w_wall[:, 1]
    grad_s[:, 1] += -w_wall[:, 2] + w_wall[:, 3]
    grad_s[:, 2] += -w_wall[:, 4] + w_wall[:, 5]
    
    w_pair = weights[n*6:]
    safe_dists = np.where(pair_dists < 1e-9, 1e-9, pair_dists)
    grads_vec = (w_pair / safe_dists)[:, np.newaxis] * diffs
    
    np.add.at(grad_s, i_idx, grads_vec)
    np.subtract.at(grad_s, j_idx, grads_vec)
    
    grad = -grad_s
    
    return obj, grad
