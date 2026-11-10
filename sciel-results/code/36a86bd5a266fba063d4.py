import numpy as np
from scipy.optimize import minimize
import time

N = 100

def construct():
    # Deadline guard
    start_time = time.time()
    max_time = 110.0  # Leave 10s margin
    
    # Initial Grid: 5x5x4
    # x: 5, y: 5, z: 4
    nx, ny, nz = 5, 5, 4
    if nx * ny * nz < N:
        # Fallback to 5x5x5 and take first N, though 100 fits exactly in 5x5x4
        nx, ny, nz = 5, 5, 5
    
    # Generate grid points in [-0.5, 0.5]
    # Spacing: 1/n
    # Positions: -0.5 + 0.5/n, -0.5 + 1.5/n, ...
    def make_ticks(n):
        return (np.arange(n) + 0.5) / n - 0.5
    
    tx = make_ticks(nx)
    ty = make_ticks(ny)
    tz = make_ticks(nz)
    
    # Create meshgrid
    X, Y, Z = np.meshgrid(tx, ty, tz, indexing='ij')
    grid = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    # Ensure we have exactly N points
    if len(grid) > N:
        grid = grid[:N]
    elif len(grid) < N:
        # Pad with random points inside (should not happen with 5x5x4=100)
        extra = np.random.rand(N - len(grid), 3) - 0.5
        grid = np.vstack([grid, extra])
        
    # Shuffle to break symmetry? 
    # Actually, keeping them in grid order helps the optimizer understand local interactions better initially?
    # Or maybe shuffling helps escape local minima.
    # Let's stick to grid order for the first optimization, then maybe shuffle if we do restarts.
    
    x0 = grid.copy()
    
    best_x = x0.copy()
    best_r = _compute_radius(best_x)
    
    # Objective function and gradient
    def objective_and_grad(x, beta):
        x = x.reshape(-1, 3)
        
        # Compute pairwise distances
        # d[i, j] = ||x_i - x_j||
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Mask diagonal (self distance)
        np.fill_diagonal(dists, np.inf)
        
        # Pairwise lengths for radius: d / 2
        # We only need i < j to avoid double counting, but for gradient symmetry it's easier to compute all and sum carefully?
        # Let's compute unique pairs.
        # Indices for i < j
        i_idx, j_idx = np.triu_indices(N, k=1)
        pair_dists = dists[i_idx, j_idx]
        
        # Wall distances
        # min_k (0.5 - |x_ik|)
        # abs_x = np.abs(x)
        # wall_dists = 0.5 - np.max(abs_x, axis=1) # This is min distance to ANY wall?
        # No, distance to nearest wall is min(0.5 - x, 0.5 + x) = 0.5 - |x|.
        # So d_i_wall = min_k (0.5 - |x_ik|).
        # This is the distance to the closest face.
        wall_dists = 0.5 - np.max(np.abs(x), axis=1)
        
        # Combine all "lengths" L
        # L consists of pair_dists/2 and wall_dists
        L = np.concatenate([pair_dists / 2.0, wall_dists])
        
        # Soft-min
        # To avoid overflow/underflow and log(0), shift L
        L_min = np.min(L)
        # If L_min is very small, beta * L might be small, exp(-beta * L) large.
        # Standard stable softmin:
        # S = -1/beta * log(sum(exp(-beta * L)))
        # Let m = min(L). sum(exp(-beta*(L-m))) * exp(-beta*m)
        # log(sum(exp(-beta*L))) = -beta*m + log(sum(exp(-beta*(L-m))))
        # S = m - 1/beta * log(sum(exp(-beta*(L-m))))
        
        # Gradient calculation requires weights w_k = exp(-beta * L_k) / sum(...)
        # Using the shifted version:
        # w_k = exp(-beta * (L_k - m)) / sum(exp(-beta * (L_j - m)))
        
        shifted = L - L_min
        exp_vals = np.exp(-beta * shifted)
        sum_exp = np.sum(exp_vals)
        weights = exp_vals / sum_exp
        
        S = L_min - (1.0 / beta) * np.log(sum_exp)
        
        # Gradient of S wrt x
        # grad_S = sum_k w_k * grad_L_k
        # Note: grad_S is for the vector x.
        # We need to accumulate gradients for each coordinate of each sphere.
        
        grad = np.zeros_like(x)
        
        # 1. Pairwise contributions
        # L_pair = dists[i,j] / 2
        # grad_{x_i} (dists[i,j]/2) = 0.5 * (x_i - x_j) / ||x_i - x_j||
        # grad_{x_j} (dists[i,j]/2) = 0.5 * (x_j - x_i) / ||x_i - x_j||
        
        # weights for pairs are weights[0 : len(pair_dists)]
        w_pairs = weights[:len(pair_dists)]
        
        # We need to add w_pairs * grad to x[i] and x[j]
        # Vectorized accumulation:
        # diff_pairs = x[i_idx] - x[j_idx]
        # unit_vecs = diff_pairs / pair_dists[:, np.newaxis]
        # grads_i = 0.5 * w_pairs[:, np.newaxis] * unit_vecs
        # grads_j = -0.5 * w_pairs[:, np.newaxis] * unit_vecs
        
        diff_pairs = x[i_idx] - x[j_idx]
        # Avoid division by zero if dist is 0 (should not happen with valid start, but just in case)
        safe_dists = np.where(pair_dists > 1e-12, pair_dists, 1.0)
        unit_vecs = diff_pairs / safe_dists[:, np.newaxis]
        
        grads_i = 0.5 * w_pairs[:, np.newaxis] * unit_vecs
        grads_j = -0.5 * w_pairs[:, np.newaxis] * unit_vecs
        
        # Accumulate
        np.add.at(grad, i_idx, grads_i)
        np.add.at(grad, j_idx, grads_j)
        
        # 2. Wall contributions
        # L_wall_i = 0.5 - max_k |x_ik|
        # Let m_i = argmax_k |x_ik|
        # grad_{x_ik} L_wall_i = -sign(x_ik) if k == m_i, else 0
        # Wait, min_k (0.5 - |x_ik|) = 0.5 - max_k |x_ik|.
        # So d/dx (0.5 - max |x|) = - d/dx (max |x|).
        # If max is at k, d/dx_k (|x_k|) = sign(x_k).
        # So grad is -sign(x_k) for the dominant coordinate.
        
        w_walls = weights[len(pair_dists):]
        
        abs_x = np.abs(x)
        max_indices = np.argmax(abs_x, axis=1)
        rows = np.arange(N)
        # grad_wall = -sign(x[rows, max_indices])
        signs = np.sign(x[rows, max_indices])
        # If x is 0, sign is 0. Gradient is undefined but 0 is safe.
        
        # Create a zero matrix and place the gradients
        # We need to add to grad[rows, max_indices]
        wall_grads = -w_walls[:, np.newaxis] * signs[:, np.newaxis]
        
        # Scatter add
        # grad[rows, max_indices] += wall_grads
        # Using np.add.at for advanced indexing accumulation
        np.add.at(grad, (rows, max_indices), wall_grads)
        
        # Objective to minimize is -S
        obj = -S
        grad_obj = -grad.ravel()
        
        return obj, grad_obj

    # Optimization Loop
    betas = [10.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    
    for beta in betas:
        if time.time() - start_time > max_time - 10:
            break
            
        # Define wrapper for scipy
        def fun(x):
            return objective_and_grad(x, beta)[0]
        
        def jac(x):
            return objective_and_grad(x, beta)[1]
            
        # Bounds
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            res = minimize(
                fun, 
                x0, 
                jac=jac, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
            )
            if res.success:
                x0 = res.x
                # Check validity and update best
                r = _compute_radius(x0)
                if r > best_r:
                    best_r = r
                    best_x = x0.copy()
        except Exception as e:
            # If optimization fails, try next beta or break
            pass
            
    # Final Polish: Use Nelder-Mead for a few steps to refine?
    # Or just return best_x
    
    return best_x

def _compute_radius(x):
    x = x.reshape(-1, 3)
    # Pairwise
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Walls
    wall_dists = 0.5 - np.max(np.abs(x), axis=1)
    min_wall = np.min(wall_dists)
    
    return min(min_pair, min_wall)
