import numpy as np
import time
import scipy.optimize as opt

N = 50

def compute_distances_and_grads(positions, alpha):
    """
    Computes the soft-min radius and its gradient.
    positions: (N, 2) array
    alpha: float
    Returns:
        soft_r: scalar
        grad: (N, 2) array
    """
    n = positions.shape[0]
    # Pairwise distances
    # d[i, j] = 0.5 * ||pos[i] - pos[j]||
    # We need min of these and min of wall distances.
    
    # Efficient computation of pairwise distances
    # diff[i, j, k] = pos[i, k] - pos[j, k]
    # This is 50x50x2, which is fine.
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    
    # Avoid division by zero for i=j, but we don't use those terms.
    # We only consider i < j.
    # Create a mask for i < j
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    # Extract distances
    dists = 0.5 * np.sqrt(dist_sq[mask])
    
    # Wall distances
    # pos[i, 0] is x, pos[i, 1] is y
    # Distances to walls: 0.5 - x, 0.5 + x, 0.5 - y, 0.5 + y
    # Since we are inside [-0.5, 0.5], 
    # d_wall_x_min = 0.5 - pos[:, 0]
    # d_wall_x_max = 0.5 + pos[:, 0]
    # d_wall_y_min = 0.5 - pos[:, 1]
    # d_wall_y_max = 0.5 + pos[:, 1]
    
    wall_dists = np.concatenate([
        0.5 - positions[:, 0],
        0.5 + positions[:, 0],
        0.5 - positions[:, 1],
        0.5 + positions[:, 1]
    ])
    
    # Combine all distances
    all_dists = np.concatenate([dists, wall_dists])
    
    # Soft min
    # S = -1/alpha * log(sum(exp(-alpha * dists)))
    # To avoid overflow/underflow, shift by max
    max_d = np.max(all_dists)
    shifted = -alpha * (all_dists - max_d)
    # Clip for stability
    shifted = np.clip(shifted, -100, 100)
    
    exp_sum = np.sum(np.exp(shifted))
    soft_r = max_d - (1.0 / alpha) * np.log(exp_sum)
    
    # Weights for gradient
    # w_i = exp(-alpha * (d_i - max_d)) / exp_sum
    weights = np.exp(shifted) / exp_sum
    
    # Gradient of soft_r w.r.t positions
    # d(soft_r)/d(pos_k) = sum_i w_i * d(d_i)/d(pos_k)
    
    grad = np.zeros_like(positions)
    
    # 1. Pairwise distances gradient
    # d(d_ij)/d(pos_k)
    # d_ij = 0.5 * ||pos_i - pos_j||
    # d(d_ij)/d(pos_i) = 0.5 * (pos_i - pos_j) / ||pos_i - pos_j|| = 0.5 * (pos_i - pos_j) / (2 * d_ij) = (pos_i - pos_j) / (4 * d_ij)
    # d(d_ij)/d(pos_j) = 0.5 * (pos_j - pos_i) / ||pos_i - pos_j|| = (pos_j - pos_i) / (4 * d_ij)
    
    # We have weights for each pair.
    # Let's reshape weights back to matrix
    w_mat = np.zeros((n, n))
    w_mat[mask] = weights[:len(dists)]
    
    # Sum contributions
    # For each pair (i, j), contribution to grad[i] is w_ij * (pos_i - pos_j) / (4 * d_ij)
    # Contribution to grad[j] is w_ij * (pos_j - pos_i) / (4 * d_ij)
    
    # To avoid division by zero, add epsilon to dists
    safe_dists = np.where(dists > 1e-10, dists, 1e-10)
    
    # Factor for each pair
    # factor_ij = w_ij / (4 * d_ij)
    # But w_mat is zero for i>=j. We need to handle symmetry.
    # Let's compute the force vector for each pair.
    
    # Re-construct full weight matrix for simplicity (N=50 is small)
    # w_full[i, j] = weights corresponding to pair (i, j) if i<j, 0 otherwise.
    # Actually, let's just iterate or use broadcasting.
    
    # grad[i] += sum_{j != i} w_ij * (pos_i - pos_j) / (4 * d_ij)
    # Note: w_ij is the weight of the distance term d_ij.
    
    # Let's create a matrix of factors F[i, j] = w_ij / (4 * d_ij) for i != j, else 0.
    # w_mat is 0 for i>=j.
    # We need w_mat[i, j] for i < j and w_mat[j, i] for j > i?
    # The soft min sum is over unique pairs.
    # So w_mat[i, j] holds the weight for pair (i, j) where i < j.
    
    # Let's make a symmetric weight matrix W where W[i, j] = W[j, i] = w_ij.
    W = w_mat + w_mat.T
    
    # Now F[i, j] = W[i, j] / (4 * d_ij)
    # d_ij is dist_sq[i, j] ** 0.5 * 0.5
    # d_ij = 0.5 * sqrt(dist_sq[i, j])
    # So 4 * d_ij = 2 * sqrt(dist_sq[i, j])
    
    # Avoid div by 0
    dist_mat = 0.5 * np.sqrt(dist_sq + 1e-12)
    dist_mat = np.where(dist_mat > 1e-10, dist_mat, 1e-10)
    
    # F[i, j] = W[i, j] / (2 * dist_mat[i, j])
    F = W / (2.0 * dist_mat)
    
    # grad[i] = sum_j F[i, j] * (pos_i - pos_j)
    # grad = F * pos_i - F * pos_j summed over j
    # grad[i] = pos_i * sum_j F[i, j] - sum_j F[i, j] * pos_j
    
    row_sums = np.sum(F, axis=1)
    weighted_pos = F @ positions
    
    grad_pairwise = positions * row_sums[:, np.newaxis] - weighted_pos
    
    # 2. Wall distances gradient
    # d(0.5 - x)/dx = -1
    # d(0.5 + x)/dx = 1
    # d(0.5 - y)/dy = -1
    # d(0.5 + y)/dy = 1
    
    # weights_wall = weights[len(dists):]
    # w_x_min, w_x_max, w_y_min, w_y_max
    w_x_min = weights[len(dists):len(dists)+n]
    w_x_max = weights[len(dists)+n:len(dists)+2*n]
    w_y_min = weights[len(dists)+2*n:len(dists)+3*n]
    w_y_max = weights[len(dists)+3*n:len(dists)+4*n]
    
    grad[:, 0] += -w_x_min + w_x_max
    grad[:, 1] += -w_y_min + w_y_max
    
    return soft_r, grad_pairwise + grad

def true_radius(positions):
    n = positions.shape[0]
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    dists = 0.5 * np.sqrt(dist_sq[mask])
    min_pair = np.min(dists)
    
    wall_dists = np.concatenate([
        0.5 - positions[:, 0],
        0.5 + positions[:, 0],
        0.5 - positions[:, 1],
        0.5 + positions[:, 1]
    ])
    min_wall = np.min(wall_dists)
    
    return min(min_pair, min_wall)

def construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    best_r = 0.0
    best_pos = None
    
    # Initializations
    # 1. Grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    init_grid = grid[:N]
    
    # 2. Random
    np.random.seed(42)
    init_random = np.random.uniform(-0.4, 0.4, (N, 2))
    
    # 3. Hexagonal-ish
    # Just use random for now, grid is a good start
    
    inits = [init_grid, init_random]
    
    alphas = [10, 50, 100, 500, 1000, 2000]
    
    for init in inits:
        current_pos = init.copy()
        current_r = true_radius(current_pos)
        if current_r > best_r:
            best_r = current_r
            best_pos = current_pos.copy()
            
        for alpha in alphas:
            if time.time() > deadline:
                break
                
            # Optimize
            def fun(x):
                pos = x.reshape(N, 2)
                r, _ = compute_distances_and_grads(pos, alpha)
                return -r
            
            def jac(x):
                pos = x.reshape(N, 2)
                _, g = compute_distances_and_grads(pos, alpha)
                return -g.flatten()
            
            try:
                res = opt.minimize(fun, current_pos.flatten(), jac=jac, method='L-BFGS-B', 
                                   bounds=[(-0.5, 0.5)] * (2 * N), options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
                new_pos = res.x.reshape(N, 2)
                new_r = true_radius(new_pos)
                
                if new_r > current_r:
                    current_pos = new_pos
                    current_r = new_r
                    if current_r > best_r:
                        best_r = current_r
                        best_pos = current_pos.copy()
            except Exception:
                pass

    # Final polish with very high alpha if time remains
    if time.time() < deadline - 5 and best_pos is not None:
        current_pos = best_pos.copy()
        current_r = true_radius(current_pos)
        for alpha in [5000, 10000]:
            if time.time() > deadline - 2:
                break
            def fun(x):
                pos = x.reshape(N, 2)
                r, _ = compute_distances_and_grads(pos, alpha)
                return -r
            
            def jac(x):
                pos = x.reshape(N, 2)
                _, g = compute_distances_and_grads(pos, alpha)
                return -g.flatten()
            
            try:
                res = opt.minimize(fun, current_pos.flatten(), jac=jac, method='L-BFGS-B', 
                                   bounds=[(-0.5, 0.5)] * (2 * N), options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
                new_pos = res.x.reshape(N, 2)
                new_r = true_radius(new_pos)
                if new_r > current_r:
                    current_pos = new_pos
                    current_r = new_r
                    if current_r > best_r:
                        best_r = current_r
                        best_pos = current_pos.copy()
            except Exception:
                pass

    if best_pos is None:
        return init_grid
        
    # Ensure validity
    # Clip to bounds just in case
    best_pos = np.clip(best_pos, -0.5, 0.5)
    
    return best_pos
