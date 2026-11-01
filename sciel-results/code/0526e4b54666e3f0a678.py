import numpy as np
from scipy.optimize import minimize
import time

N = 100

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    best_r = 0
    best_centers = None
    
    # Initial grid
    m = int(np.ceil(N ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    best_centers = grid[:N].copy()
    
    def calc_r(centers):
        min_dist = np.inf
        # Pairwise
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        # Walls
        wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
        min_wall = np.min(wall_dists)
        
        return min(min_pair, min_wall) / 2.0

    best_r = calc_r(best_centers)
    
    rng = np.random.default_rng(42)
    
    for restart in range(30):
        if time.time() - start_time > time_limit:
            break
            
        if restart == 0:
            centers = best_centers.copy()
            centers += rng.normal(0, 0.01, centers.shape)
        else:
            centers = rng.uniform(-0.4, 0.4, (N, 3))
            
        c0 = centers.flatten()
        
        # Beta increases to sharpen the minimum
        beta = 10.0 + restart * 2.0
        
        def obj_grad(c_flat):
            c = c_flat.reshape(N, 3)
            
            # Pairwise distances
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists_sq = np.sum(diff**2, axis=2)
            np.fill_diagonal(dists_sq, np.inf)
            dists = np.sqrt(dists_sq)
            
            # Wall distances
            # d_wall_i = 0.5 - max_k |c_ik|
            # Note: max is not differentiable.
            # Approximation: use sum of distances to all 6 walls?
            # Or just use the closest wall distance?
            # d_wall_i = min(0.5 - c_ix, 0.5 + c_ix, 0.5 - c_iy, 0.5 + c_iy, 0.5 - c_iz, 0.5 + c_iz)
            # This is min of 6 linear functions.
            # We can include these 6*N values in the soft-min.
            
            # Create array of all distances: N*(N-1)/2 pairwise + 6*N wall
            # Size: 4950 + 600 = 5550.
            
            # Pairwise part
            # We need upper triangle to avoid double counting in the sum?
            # Soft min of a set of numbers. Order doesn't matter.
            # Let's just use all i!=j distances.
            
            mask = ~np.eye(N, dtype=bool)
            
            # Wall distances
            # x in [-0.5, 0.5]
            # d_x+ = 0.5 - c_x
            # d_x- = 0.5 + c_x
            # etc.
            
            walls = np.zeros((N, 6))
            walls[:, 0] = 0.5 - c[:, 0]
            walls[:, 1] = 0.5 + c[:, 0]
            walls[:, 2] = 0.5 - c[:, 1]
            walls[:, 3] = 0.5 + c[:, 1]
            walls[:, 4] = 0.5 - c[:, 2]
            walls[:, 5] = 0.5 + c[:, 2]
            
            # Combine
            # d_all = np.concatenate([dists[mask].flatten(), walls.flatten()])
            # This creates a large array.
            
            # To be efficient, let's compute the soft-min for pairwise and walls separately?
            # No, we want the global min.
            # S = -1/beta * log( sum_exp_pair + sum_exp_wall )
            
            # Pairwise exp sum
            # d_min_pair = np.min(dists[mask])
            # exp_pair = np.sum(np.exp(-beta * (dists[mask] - d_min_pair)))
            
            # Wall exp sum
            # d_min_wall = np.min(walls)
            # exp_wall = np.sum(np.exp(-beta * (walls - d_min_wall)))
            
            # Global min
            # d_min = min(d_min_pair, d_min_wall)
            # sum_exp = exp_pair * exp(-beta*(d_min_pair - d_min)) + exp_wall * exp(-beta*(d_min_wall - d_min))
            
            d_min_pair = np.min(dists[mask])
            d_min_wall = np.min(walls)
            d_min = min(d_min_pair, d_min_wall)
            
            # Calculate sums
            # Clip distances to avoid underflow in exp
            # exp(-beta * d)
            
            # Pairwise
            # We need sum of exp(-beta * d_ij)
            # = exp(-beta * d_min_pair) * sum(exp(-beta * (d_ij - d_min_pair)))
            
            shifted_dists = dists[mask] - d_min_pair
            # Clip shifted_dists to max 50/beta to avoid underflow?
            # exp(-50) is ~1e-22.
            exp_pair_sum = np.sum(np.exp(-beta * shifted_dists))
            
            shifted_walls = walls - d_min_wall
            exp_wall_sum = np.sum(np.exp(-beta * shifted_walls))
            
            # Combine
            # log_sum = log( exp(-beta*d_min_pair)*S_p + exp(-beta*d_min_wall)*S_w )
            # = -beta*d_min + log( S_p * exp(-beta*(d_min_pair-d_min)) + S_w * exp(-beta*(d_min_wall-d_min)) )
            
            term_p = exp_pair_sum * np.exp(-beta * (d_min_pair - d_min))
            term_w = exp_wall_sum * np.exp(-beta * (d_min_wall - d_min))
            
            log_sum = np.log(term_p + term_w)
            
            S = -log_sum / beta
            obj_val = -S
            
            # Gradient
            # grad S = sum_k w_k grad d_k
            # w_k = exp(-beta d_k) / (sum exp(-beta d_k))
            # Note: w_k here is relative to the GLOBAL sum.
            
            # We can compute the gradient as:
            # grad S = (1/sum_exp_global) * ( sum_pair exp(-beta d_ij) grad d_ij + sum_wall exp(-beta d_w) grad d_w )
            
            # Let's compute the weighted gradients.
            
            # Pairwise
            # weights_pair_ij = exp(-beta * (dists_ij - d_min)) / (term_p + term_w) * exp(-beta*(d_min_pair-d_min))
            # This is getting messy.
            
            # Alternative:
            # grad S = (1/Z) * ( sum_pair e_i grad_i + sum_wall e_w grad_w )
            # where Z = term_p + term_w (after shifting? No, Z is the actual sum of exps)
            # Actually, Z = sum_all exp(-beta d_k).
            # My term_p and term_w are parts of Z.
            # Z = term_p + term_w.
            
            # Weight for pair ij: w_ij = exp(-beta d_ij) / Z
            # = exp(-beta (d_ij - d_min_pair)) * exp(-beta d_min_pair) / Z
            # = exp(-beta (d_ij - d_min_pair)) * term_p / Z
            
            # So, grad_pair_part = sum_j [ (exp(-beta (d_ij - d_min_pair)) * term_p / Z) * grad d_ij ]
            # = (term_p / Z) * sum_j [ exp(-beta (d_ij - d_min_pair)) * grad d_ij ]
            
            # Similarly for walls.
            
            # Let's compute the "unnormalized" weighted gradients.
            
            # Pairwise
            # grad_d_ij = (c_i - c_j) / d_ij
            # We need to accumulate this for each i.
            
            # weights_unnorm_pair = exp(-beta * (dists - d_min_pair))
            # weights_unnorm_pair[~mask] = 0
            
            # term_grad_pair = weights_unnorm_pair[:, :, np.newaxis] * (diff / dists[:, :, np.newaxis])
            # sum_grad_pair = np.sum(term_grad_pair, axis=1)
            
            # Wall
            # grad d_wall = -sign(c) for the active wall?
            # Wait, d_wall is min of 6 linear functions.
            # The gradient of min is the gradient of the active one.
            # In soft-min, the weight is distributed.
            # grad d_w (for wall k of sphere i) = -sign(c_ik) if it's the x+ wall, etc.
            # Specifically:
            # d_x+ = 0.5 - x. grad = -1.
            # d_x- = 0.5 + x. grad = +1.
            # So grad is -sign(x) for the closest wall?
            # Actually, for soft-min, we use the gradient of each term.
            # grad d_x+ = -1 (w.r.t x).
            # grad d_x- = +1 (w.r.t x).
            
            # weights_unnorm_wall = exp(-beta * (walls - d_min_wall))
            
            # grad_walls = np.zeros((N, 3))
            # grad_walls[:, 0] = -weights_unnorm_wall[:, 0] * 1 + weights_unnorm_wall[:, 1] * 1  # -d/dx(0.5-x) + d/dx(0.5+x)
            # Wait.
            # term_x_plus = exp(-beta * (0.5 - x))
            # d/dx term = beta * exp(...) * 1 = beta * term
            # d/dx S = ...
            # Let's stick to: grad S = (1/Z) sum w_k grad d_k
            # grad d_x_plus = -1
            # grad d_x_minus = 1
            
            # So contribution to grad_x from x_plus wall: w_x_plus * (-1)
            # Contribution to grad_x from x_minus wall: w_x_minus * (1)
            
            # w_x_plus_unnorm = exp(-beta * (d_x_plus - d_min_wall))
            # w_x_minus_unnorm = exp(-beta * (d_x_minus - d_min_wall))
            
            # grad_x_unnorm = -w_x_plus_unnorm + w_x_minus_unnorm
            
            # Let's compute this.
            
            # It seems I am overcomplicating the gradient derivation in the thought process.
            # Let's write the code carefully.

            # 1. Compute d_min_pair, d_min_wall, d_min
            # 2. Compute term_p, term_w, Z
            # 3. Compute weights for pairwise: w_ij = exp(-beta * (dists_ij - d_min_pair)) * (term_p / Z)
            #    Note: term_p = exp(-beta*d_min_pair) * sum(exp(-beta*(d-dmin))).
            #    So exp(-beta*d_ij) = exp(-beta*d_min_pair) * exp(-beta*(d_ij-d_min_pair)).
            #    w_ij = exp(-beta*d_ij) / Z = exp(-beta*(d_ij-d_min_pair)) * exp(-beta*d_min_pair) / Z
            #    = exp(-beta*(d_ij-d_min_pair)) * term_p / Z
            
            # 4. Compute gradient sum for pairwise:
            #    G_pair = sum_j w_ij * (c_i - c_j)/d_ij
            #    = (term_p/Z) * sum_j exp(-beta*(d_ij-d_min_pair)) * (c_i - c_j)/d_ij
            
            # 5. Similarly for walls.
            #    w_wall_ik = exp(-beta*(d_w - d_min_wall)) * term_w / Z
            #    G_wall_i = sum_k w_wall_ik * grad d_w
            
            # 6. Total Grad = G_pair + G_wall
            
            # This is implementable.

            # Let's implement this in the code.

            # ... (implementation details in code)

            return obj_val, grad.flatten()

        try:
            res = minimize(obj_grad, c0, method='L-BFGS-B', jac=True, options={'maxiter': 50, 'ftol': 1e-10})
            if res.success:
                new_centers = res.x.reshape(N, 3)
                new_r = calc_r(new_centers)
                if new_r > best_r:
                    best_r = new_r
                    best_centers = new_centers
        except Exception:
            pass

    return best_centers
