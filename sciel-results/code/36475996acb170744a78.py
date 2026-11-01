import numpy as np
from scipy.optimize import minimize

N = 50

def construct():
    best_score = -np.inf
    best_centers = None
    
    def get_distances(centers):
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(N, k=1)
        d_pairs = dists[iu]
        
        # Wall distances
        abs_centers = np.abs(centers)
        max_abs = np.max(abs_centers, axis=1)
        d_walls = 0.5 - max_abs
        
        return d_pairs, d_walls

    def obj_and_grad(params, beta):
        centers = params.reshape(N, 3)
        d_pairs, d_walls = get_distances(centers)
        
        # Combine all distances for log-sum-exp
        # f = -1/beta * log(sum(exp(-beta * d)))
        # Stable log-sum-exp
        z_pairs = -beta * d_pairs
        z_walls = -beta * d_walls
        
        # Calculate log(S_pairs)
        max_z_pairs = np.max(z_pairs)
        if np.isinf(max_z_pairs):
            log_s_pairs = max_z_pairs
        else:
            log_s_pairs = max_z_pairs + np.log(np.sum(np.exp(z_pairs - max_z_pairs)))
            
        # Calculate log(S_walls)
        max_z_walls = np.max(z_walls)
        if np.isinf(max_z_walls):
            log_s_walls = max_z_walls
        else:
            log_s_walls = max_z_walls + np.log(np.sum(np.exp(z_walls - max_z_walls)))
            
        # Combine
        m = np.maximum(log_s_pairs, log_s_walls)
        log_s = m + np.log(np.exp(log_s_pairs - m) + np.exp(log_s_walls - m))
        
        val = -1.0 / beta * log_s
        
        # Gradient of f
        # df/dc = sum w_k dd_k/dc
        # w_k = exp(-beta d_k) / S = exp(z_k - log_s)
        
        w_pairs = np.exp(z_pairs - log_s)
        w_walls = np.exp(z_walls - log_s)
        
        grad_params = np.zeros_like(centers)
        
        # Pairwise
        iu = np.triu_indices(N, k=1)
        for idx in range(len(iu[0])):
            i, j = iu[0][idx], iu[1][idx]
            w = w_pairs[idx]
            if w < 1e-10:
                continue
            d = d_pairs[idx]
            if d < 1e-6:
                # Avoid division by zero
                # If d is very small, gradient is huge.
                # Use a small epsilon
                d = 1e-6
            
            diff_ij = centers[i] - centers[j]
            # df/dc_i += w * (c_i - c_j) / d
            # df/dc_j += w * (c_j - c_i) / d
            grad_vec = w * diff_ij / d
            grad_params[i] += grad_vec
            grad_params[j] -= grad_vec
            
        # Walls
        # d_w_i = 0.5 - max(|x|, |y|, |z|)
        # d(d_w)/dc = - d(max_abs)/dc
        # d(max_abs)/dc_k = sign(c_k) * indicator(c_k is max)
        # In the soft version (log-sum-exp), the "indicator" is replaced by weights.
        # But here we used hard max for d_walls?
        # NO, I used `d_walls = 0.5 - max_abs`. This is NOT differentiable at the max.
        # The log-sum-exp `f` uses `d_walls`.
        # If `d_walls` is non-differentiable, the gradient is wrong.
        # I need to use a smooth approximation for the wall distance too?
        # Or, since we are maximizing min(d_pairs, d_walls), and d_walls is min of 3 distances to walls,
        # we can include all 3*N wall distances in the log-sum-exp sum.
        
        # Let's refactor to include all wall distances explicitly.
        # d_w_ik = 0.5 - sign_k * c_ik? No.
        # Distance to wall x=0.5 is 0.5 - x.
        # Distance to wall x=-0.5 is x + 0.5.
        # We want min of these 3 for each axis? No, min of all 6 for each sphere?
        # Yes, distance to nearest wall.
        
        # Let's recompute d_walls as a vector of 3*N values?
        # Or 6*N values?
        # If we use 6*N values, the log-sum-exp will naturally pick the smallest ones.
        
        # Let's restart the distance calculation for walls.
        # For each sphere i:
        # d_x_plus = 0.5 - x_i
        # d_x_minus = 0.5 + x_i
        # d_y_plus = 0.5 - y_i
        # d_y_minus = 0.5 + y_i
        # d_z_plus = 0.5 - z_i
        # d_z_minus = 0.5 + z_i
        
        # We can create a matrix of shape (N, 6).
        
        return val, grad_params.flatten()

    # The above `obj_and_grad` is incomplete because I realized `d_walls` using `max` is not smooth.
    # I will rewrite `obj_and_grad` to use all 6 wall distances per sphere.
    
    def obj_and_grad_v2(params, beta):
        centers = params.reshape(N, 3)
        
        # Pairwise
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(N, k=1)
        d_pairs = dists[iu]
        
        # Walls: 6 distances per sphere
        # +x, -x, +y, -y, +z, -z
        d_walls = np.empty((N, 6))
        d_walls[:, 0] = 0.5 - centers[:, 0]
        d_walls[:, 1] = 0.5 + centers[:, 0]
        d_walls[:, 2] = 0.5 - centers[:, 1]
        d_walls[:, 3] = 0.5 + centers[:, 1]
        d_walls[:, 4] = 0.5 - centers[:, 2]
        d_walls[:, 5] = 0.5 + centers[:, 2]
        
        d_walls_flat = d_walls.flatten()
        
        # All distances
        # We can concatenate or handle separately.
        # Handling separately is cleaner for gradient accumulation.
        
        z_pairs = -beta * d_pairs
        z_walls = -beta * d_walls_flat
        
        max_z_pairs = np.max(z_pairs)
        log_s_pairs = max_z_pairs + np.log(np.sum(np.exp(z_pairs - max_z_pairs)))
        
        max_z_walls = np.max(z_walls)
        log_s_walls = max_z_walls + np.log(np.sum(np.exp(z_walls - max_z_walls)))
        
        m = np.maximum(log_s_pairs, log_s_walls)
        log_s = m + np.log(np.exp(log_s_pairs - m) + np.exp(log_s_walls - m))
        
        val = -1.0 / beta * log_s
        
        w_pairs = np.exp(z_pairs - log_s)
        w_walls = np.exp(z_walls - log_s)
        
        grad_params = np.zeros_like(centers)
        
        # Pairwise
        for idx in range(len(iu[0])):
            i, j = iu[0][idx], iu[1][idx]
            w = w_pairs[idx]
            if w < 1e-10: continue
            d = d_pairs[idx]
            if d < 1e-6: d = 1e-6
            diff_ij = centers[i] - centers[j]
            grad_vec = w * diff_ij / d
            grad_params[i] += grad_vec
            grad_params[j] -= grad_vec
            
        # Walls
        # d_w_ik is distance to wall k of sphere i.
        # Gradient of d_w_ik wrt c_i:
        # +x (0.5 - x): -1
        # -x (0.5 + x): +1
        # +y (0.5 - y): -1
        # -y (0.5 + y): +1
        # +z (0.5 - z): -1
        # -z (0.5 + z): +1
        
        # grad_params[i] += sum_k w_ik * grad(d_ik)/dc_i
        # grad(d_ik)/dc_i is a vector.
        # For k=0 (+x): [ -1, 0, 0 ]
        # For k=1 (-x): [ +1, 0, 0 ]
        # For k=2 (+y): [ 0, -1, 0 ]
        # For k=3 (-y): [ 0, +1, 0 ]
        # For k=4 (+z): [ 0, 0, -1 ]
        # For k=5 (-z): [ 0, 0, +1 ]
        
        # We can accumulate this efficiently.
        # grad_params[i, 0] += w_0 * (-1) + w_1 * (1)
        # grad_params[i, 1] += w_2 * (-1) + w_3 * (1)
        # grad_params[i, 2] += w_4 * (-1) + w_5 * (1)
        
        w_walls_reshaped = w_walls.reshape(N, 6)
        
        # Vectorized update
        # grad_x = -w[:,0] + w[:,1]
        # grad_y = -w[:,2] + w[:,3]
        # grad_z = -w[:,4] + w[:,5]
        
        grad_params[:, 0] += -w_walls_reshaped[:, 0] + w_walls_reshaped[:, 1]
        grad_params[:, 1] += -w_walls_reshaped[:, 2] + w_walls_reshaped[:, 3]
        grad_params[:, 2] += -w_walls_reshaped[:, 4] + w_walls_reshaped[:, 5]
        
        return val, grad_params.flatten()

    def optimize_start(centers_init):
        centers = centers_init.copy()
        betas = [5.0, 20.0, 50.0, 100.0, 200.0]
        
        for beta in betas:
            def obj(p):
                return -obj_and_grad_v2(p, beta)[0]
            
            def grad(p):
                return -obj_and_grad_v2(p, beta)[1]
            
            res = minimize(
                obj,
                centers.flatten(),
                jac=grad,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 100, 'gtol': 1e-5}
            )
            centers = res.x.reshape(N, 3)
            
        return centers

    best_c = None
    best_r = -1
    
    for k in range(10):
        np.random.seed(100 + k)
        c_init = np.random.uniform(-0.45, 0.45, (N, 3))
        c_opt = optimize_start(c_init)
        
        diff = c_opt[:, np.newaxis, :] - c_opt[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        max_c = np.max(np.abs(c_opt), axis=1)
        min_wall = 0.5 - np.min(max_c)
        
        r = min(min_pair, min_wall)
        
        if r > best_r:
            best_r = r
            best_c = c_opt
            
    return best_c
