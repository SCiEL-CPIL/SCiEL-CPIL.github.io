"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize

N = 100

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Pre-compute pair indices
    i_idx, j_idx = np.triu_indices(N, k=1)
    n_pairs = len(i_idx)
    
    # Wall indices: for each sphere i, 6 walls
    # Wall 0: x < -0.5 -> dist = c_x + 0.5
    # Wall 1: x > 0.5 -> dist = 0.5 - c_x
    # Wall 2: y < -0.5 -> dist = c_y + 0.5
    # Wall 3: y > 0.5 -> dist = 0.5 - c_y
    # Wall 4: z < -0.5 -> dist = c_z + 0.5
    # Wall 5: z > 0.5 -> dist = 0.5 - c_z
    
    # Function to compute the vector of all distances X
    def get_distances(pos):
        # Pairwise
        diff = pos[i_idx] - pos[j_idx]
        dists_pair = 0.5 * np.linalg.norm(diff, axis=1)
        
        # Walls
        # pos: (N, 3)
        # dists_wall: (N, 6)
        # col 0: pos[:, 0] + 0.5
        # col 1: 0.5 - pos[:, 0]
        # col 2: pos[:, 1] + 0.5
        # col 3: 0.5 - pos[:, 1]
        # col 4: pos[:, 2] + 0.5
        # col 5: 0.5 - pos[:, 2]
        dists_wall = np.empty((N, 6))
        dists_wall[:, 0] = pos[:, 0] + 0.5
        dists_wall[:, 1] = 0.5 - pos[:, 0]
        dists_wall[:, 2] = pos[:, 1] + 0.5
        dists_wall[:, 3] = 0.5 - pos[:, 1]
        dists_wall[:, 4] = pos[:, 2] + 0.5
        dists_wall[:, 5] = 0.5 - pos[:, 2]
        
        # Flatten
        X = np.concatenate([dists_pair, dists_wall.flatten()])
        return X, diff, dists_pair

    def soft_min_value_and_grad(pos, beta):
        X, diff, dists_pair = get_distances(pos)
        
        # Shift for stability
        x_min = np.min(X)
        exp_terms = np.exp(-beta * (X - x_min))
        sum_exp = np.sum(exp_terms)
        
        # Value: soft_min = x_min - (1/beta) * log(sum_exp)
        val = x_min - (1.0 / beta) * np.log(sum_exp)
        
        # We want to maximize val, so minimize -val.
        # Return -val and -grad_val
        
        # Weights
        weights = exp_terms / sum_exp
        
        grad_val = np.zeros_like(pos)
        
        # Pair contributions
        # d x_k / d p_i = 0.5 * (p_i - p_j) / ||p_i - p_j||
        # d x_k / d p_j = 0.5 * (p_j - p_i) / ||p_i - p_j||
        # x_k = 0.5 * norm
        # norm = 2 * dists_pair
        norms = 2.0 * dists_pair
        norms_safe = np.where(norms < 1e-8, 1e-8, norms)
        
        w_pair = weights[:n_pairs]
        # coeff for grad
        # grad_i += w * 0.5 * diff / norm
        # = w * diff / (2 * norm)
        # = w * diff / (4 * dists_pair)
        coeff = w_pair / (4.0 * dists_pair)
        coeff = np.where(dists_pair < 1e-8, 0.0, coeff)
        
        grad_vecs = coeff[:, np.newaxis] * diff
        
        np.add.at(grad_val, i_idx, grad_vecs)
        np.add.at(grad_val, j_idx, -grad_vecs)
        
        # Wall contributions
        # X indices for walls start at n_pairs
        # w_wall = weights[n_pairs:]
        # Shape (N, 6)
        w_wall = weights[n_pairs:].reshape(N, 6)
        
        # d x / d c
        # Wall 0 (c_x + 0.5): d/dc_x = 1
        # Wall 1 (0.5 - c_x): d/dc_x = -1
        # Wall 2 (c_y + 0.5): d/dc_y = 1
        # Wall 3 (0.5 - c_y): d/dc_y = -1
        # Wall 4 (c_z + 0.5): d/dc_z = 1
        # Wall 5 (0.5 - c_z): d/dc_z = -1
        
        # grad_val[:, 0] += w_wall[:, 0] * 1 + w_wall[:, 1] * (-1)
        # grad_val[:, 0] += w_wall[:, 0] - w_wall[:, 1]
        # grad_val[:, 1] += w_wall[:, 2] - w_wall[:, 3]
        # grad_val[:, 2] += w_wall[:, 4] - w_wall[:, 5]
        
        grad_val[:, 0] += w_wall[:, 0] - w_wall[:, 1]
        grad_val[:, 1] += w_wall[:, 2] - w_wall[:, 3]
        grad_val[:, 2] += w_wall[:, 4] - w_wall[:, 5]
        
        # We minimize -val, so return -val and -grad_val
        return -val, -grad_val

    def objective(pos, beta):
        val, grad = soft_min_value_and_grad(pos, beta)
        return val, grad

    # Initial Grid
    def make_grid(nx, ny, nz):
        xs = (np.arange(nx) + 0.5) / nx - 0.5
        ys = (np.arange(ny) + 0.5) / ny - 0.5
        zs = (np.arange(nz) + 0.5) / nz - 0.5
        grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
        return grid[:N]
    
    base_grid = make_grid(5, 5, 4)
    
    best_pos = base_grid.copy()
    best_score = -np.inf
    
    # Optimization Loop
    num_restarts = 5
    for i in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Perturb
        pos = base_grid + np.random.normal(0, 0.01, base_grid.shape)
        # Clip to bounds
        pos = np.clip(pos, -0.5 + 1e-6, 0.5 - 1e-6)
        
        # Stage 1: Smooth
        # beta = 10
        res1 = scipy.optimize.minimize(
            lambda p: objective(p, 10.0)[0],
            pos,
            jac=lambda p: objective(p, 10.0)[1],
            method='L-BFGS-B',
            bounds=[(-0.5 + 1e-6, 0.5 - 1e-6)] * (N * 3),
            options={'maxiter': 100, 'ftol': 1e-6}
        )
        
        if time.time() > deadline - 5.0:
             if res1.fun < best_score:
                 best_score = res1.fun
                 best_pos = res1.x.reshape(N, 3)
             break

        # Stage 2: Sharper
        # beta = 50
        pos2 = res1.x.reshape(N, 3)
        res2 = scipy.optimize.minimize(
            lambda p: objective(p, 50.0)[0],
            pos2,
            jac=lambda p: objective(p, 50.0)[1],
            method='L-BFGS-B',
            bounds=[(-0.5 + 1e-6, 0.5 - 1e-6)] * (N * 3),
            options={'maxiter': 200, 'ftol': 1e-8}
        )
        
        # Calculate actual min radius for this configuration
        X, _, _ = get_distances(res2.x.reshape(N, 3))
        r = np.min(X)
        
        # The objective value is -soft_min.
        # We want to maximize r.
        # Compare r directly.
        if r > best_score:
            best_score = r
            best_pos = res2.x.reshape(N, 3)

    # Final check: ensure best_pos is valid
    # If best_pos is still the initial grid (score -inf), use it.
    # But we initialized best_score to -inf.
    # If no optimization ran, best_pos is base_grid.
    
    # One more polish with high beta if time permits
    if time.time() < deadline - 2.0:
         pos3 = best_pos
         res3 = scipy.optimize.minimize(
            lambda p: objective(p, 100.0)[0],
            pos3,
            jac=lambda p: objective(p, 100.0)[1],
            method='L-BFGS-B',
            bounds=[(-0.5 + 1e-6, 0.5 - 1e-6)] * (N * 3),
            options={'maxiter': 100, 'ftol': 1e-10}
        )
         X, _, _ = get_distances(res3.x.reshape(N, 3))
         r = np.min(X)
         if r > best_score:
             best_score = r
             best_pos = res3.x.reshape(N, 3)

    return best_pos

# EVOLVE-BLOCK-END
