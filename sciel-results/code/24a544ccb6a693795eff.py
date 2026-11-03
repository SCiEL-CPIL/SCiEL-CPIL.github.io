import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initialization: Simple grid
    # 4x4x4 grid = 64 points. Take 50.
    # To make it more isotropic, maybe shuffle?
    # Or just take a subset that is well distributed.
    # Let's use a 3x4x5 grid? 60 points.
    # Or just 4x4x4.
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # Randomly select 50? Or just first 50?
    # First 50 might be biased towards one corner.
    # Let's shuffle and take 50.
    np.random.seed(42)
    idx = np.random.choice(len(grid), N, replace=False)
    X = grid[idx].copy()
    
    # Optimization stages
    betas = [10.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    
    best_X = X
    best_score = -np.inf
    
    for beta in betas:
        if time.time() > deadline:
            break
            
        def objective_and_grad(X_flat):
            X = X_flat.reshape(N, 3)
            # Compute pairwise distances
            # Use broadcasting
            diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            # Mask upper triangle
            mask = np.triu(np.ones((N, N), dtype=bool), k=1)
            dists = np.sqrt(dist_sq[mask])
            
            # Compute wall distances
            # Left/Right for x, y, z
            # d_L = x + 0.5, d_R = 0.5 - x
            wall_dists = np.concatenate([
                X[:, 0] + 0.5, 0.5 - X[:, 0],
                X[:, 1] + 0.5, 0.5 - X[:, 1],
                X[:, 2] + 0.5, 0.5 - X[:, 2]
            ])
            
            D = np.concatenate([dists, wall_dists])
            
            # Soft min
            # f = -1/beta * log(sum(exp(-beta * D)))
            # To avoid overflow/underflow, use log-sum-exp trick
            # log(sum(exp(-beta * D))) = -beta * max(D) + log(sum(exp(-beta * (D - max(D)))))
            max_D = np.max(D)
            log_sum_exp = -beta * max_D + np.log(np.sum(np.exp(-beta * (D - max_D))))
            f = -log_sum_exp / beta
            
            # Gradient
            # df/dX = sum_k W_k * grad_d_k
            # W_k = exp(-beta * D_k) / sum(exp(-beta * D_j))
            # Note: sum(exp(-beta * D_j)) = exp(log_sum_exp)
            # So W_k = exp(-beta * (D_k - max_D)) / sum(exp(-beta * (D_j - max_D)))
            exp_terms = np.exp(-beta * (D - max_D))
            sum_exp = np.sum(exp_terms)
            W = exp_terms / sum_exp
            
            grad = np.zeros((N, 3))
            
            # Pairwise gradients
            # W_pair is the first binom(N,2) elements of W
            W_pair = W[:len(dists)]
            
            # We need to map W_pair back to the (i, j) pairs
            # dists was extracted using mask
            # We can compute the gradient contribution directly
            # grad_i += W_ij * (x_i - x_j) / d_ij
            # grad_j -= W_ij * (x_i - x_j) / d_ij
            
            # Vectorized approach:
            # Create a matrix of weights W_mat (N x N)
            W_mat = np.zeros((N, N))
            W_mat[mask] = W_pair
            
            # diff is x_i - x_j
            # dist is d_ij
            # We need d_ij in the same shape as diff
            dist_mat = np.sqrt(dist_sq)
            # Avoid division by zero
            dist_mat_safe = np.where(dist_sq > 1e-12, dist_mat, 1.0)
            
            # grad contribution from pairs
            # term = W_mat * (diff / dist_mat_safe[:, :, np.newaxis])
            # grad_i = sum_j term_ij
            # grad_j = -sum_i term_ij (since term_ji = -term_ij and W_ji = W_ij)
            
            # Actually, W_mat is symmetric.
            # grad_X = sum_j W_mat * (X_i - X_j) / d_ij
            # This is equivalent to:
            # G = W_mat / dist_mat_safe
            # grad_X = sum_j G * (X_i - X_j)
            # = sum_j G * X_i - sum_j G * X_j
            # = (sum_j G) * X_i - sum_j (G * X_j)
            
            G = W_mat / dist_mat_safe
            sum_G = np.sum(G, axis=1) # Shape (N,)
            # sum_j (G_ij * X_j) -> Matrix multiply G.T @ X?
            # G is N x N. X is N x 3.
            # (G @ X) is N x 3. Element i is sum_j G_ij X_j.
            # So grad_pair = sum_G[:, np.newaxis] * X - (G @ X)
            
            grad_pair = sum_G[:, np.newaxis] * X - (G @ X)
            
            # Wall gradients
            W_wall = W[len(dists):]
            # Wall dists are ordered:
            # 0..N-1: x_L (x+0.5), grad x: 1
            # N..2N-1: x_R (0.5-x), grad x: -1
            # 2N..3N-1: y_L, grad y: 1
            # 3N..4N-1: y_R, grad y: -1
            # 4N..5N-1: z_L, grad z: 1
            # 5N..6N-1: z_R, grad z: -1
            
            grad_wall = np.zeros((N, 3))
            grad_wall[:, 0] += W_wall[0:N]
            grad_wall[:, 0] -= W_wall[N:2*N]
            grad_wall[:, 1] += W_wall[2*N:3*N]
            grad_wall[:, 1] -= W_wall[3*N:4*N]
            grad_wall[:, 2] += W_wall[4*N:5*N]
            grad_wall[:, 2] -= W_wall[5*N:6*N]
            
            total_grad = grad_pair + grad_wall
            
            # We want to maximize f, so minimize -f
            # grad of -f is -total_grad
            return -f, -total_grad.flatten()

        # Run optimization
        X_flat = X.flatten()
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            res = minimize(
                objective_and_grad,
                X_flat,
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.success or res.fun < -best_score: # Note: fun is -f, so lower is better for -f?
                # Wait, objective returns -f. So we want to minimize -f.
                # So res.fun is -f_max.
                # We want to maximize f. So we want to minimize res.fun.
                # best_score is max f.
                # So if -res.fun > best_score, i.e., res.fun < -best_score.
                if -res.fun > best_score:
                    best_score = -res.fun
                    best_X = res.x.reshape(N, 3)
        except Exception:
            pass

    # Final polish: Compute exact radius
    # The score is the min distance.
    # We can do a few iterations of a simpler local search or just return best_X.
    # The soft-min with high beta should be very close to the true min.
    
    return best_X
