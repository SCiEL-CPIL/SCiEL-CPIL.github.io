import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Initialization: Jittered grid
    # 4x4x4 grid has 64 points. We need 50.
    # Let's create a 4x4x4 grid in [-0.45, 0.45] to avoid boundary issues initially.
    m = 4
    # Spacing is 0.9 / 3 = 0.3? No, we want to cover the cube.
    # Let's just use random points for simplicity, or a simple grid.
    # Random points are often okay with soft-min.
    np.random.seed(42)
    C = np.random.uniform(-0.45, 0.45, (N, 3))
    
    best_C = C.copy()
    best_radius = -1.0
    
    def compute_radius(C):
        # Compute min pairwise distance / 2
        # and min distance to boundary
        # Vectorized pairwise distance
        # C is (N, 3)
        diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Set diagonal to inf
        np.fill_diagonal(dists, np.inf)
        min_pairwise = np.min(dists) / 2.0
        
        # Boundary
        # Distance to closest wall for each point
        # x in [-0.5, 0.5]
        # dist_x = 0.5 - abs(C[:, 0])
        # dist_y = 0.5 - abs(C[:, 1])
        # dist_z = 0.5 - abs(C[:, 2])
        d_b = 0.5 - np.abs(C)
        min_boundary = np.min(d_b)
        
        return min(min_pairwise, min_boundary)

    best_radius = compute_radius(best_C)
    
    # Annealing parameters
    betas = np.linspace(10, 1000, 20)
    
    def soft_min_obj_and_grad(C_flat, beta):
        C = C_flat.reshape(N, 3)
        
        # Compute all distances x_k
        # 1. Pairwise
        diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        # We want d_ij / 2
        # Create a list of pairs i < j
        i_idx, j_idx = np.triu_indices(N, k=1)
        pair_dists = dists[i_idx, j_idx] / 2.0
        
        # 2. Boundary
        # For each point i and axis a, dist = 0.5 - abs(C[i, a])
        # We treat each (i, a) as a separate term in the soft min
        # There are N*3 such terms
        d_b = 0.5 - np.abs(C)
        # Flatten d_b to get individual terms
        boundary_dists = d_b.flatten()
        
        # Combine all distances
        all_dists = np.concatenate([pair_dists, boundary_dists])
        
        # Soft min: -1/beta * log(sum(exp(-beta * x)))
        # For numerical stability, subtract max
        max_d = np.max(all_dists)
        # Note: we want to maximize min, so we minimize -soft_min.
        # soft_min = -1/beta * log(sum(exp(-beta * x)))
        # Let y = -beta * x
        # sum(exp(y))
        # To avoid overflow/underflow:
        # log(sum(exp(y))) = max_y + log(sum(exp(y - max_y)))
        
        y = -beta * all_dists
        max_y = np.max(y)
        if not np.isfinite(max_y):
             return 0.0, np.zeros_like(C_flat)
        
        exp_y = np.exp(y - max_y)
        sum_exp = np.sum(exp_y)
        log_sum = max_y + np.log(sum_exp)
        
        obj = -1.0 / beta * log_sum
        
        # Gradient
        # d(obj)/d(x_k) = (1/beta) * (exp(-beta*x_k) / sum(exp(-beta*x_j)))
        # Note: exp_y[k] = exp(-beta*x_k - max_y)
        # sum_exp = sum(exp_y)
        # weights w_k = exp_y[k] / sum_exp
        # d(obj)/d(x_k) = (1/beta) * w_k
        
        w = exp_y / sum_exp
        dw_dx = w / beta
        
        # Now map dw_dx back to C
        grad_C = np.zeros_like(C)
        
        # 1. Pairwise terms
        # w_pair = dw_dx[:len(pair_dists)]
        w_pair = dw_dx[:len(pair_dists)]
        
        # grad_C[i] += sum_{j>i} w_pair[k] * (C[i] - C[j]) / (2 * dist_ij)
        # grad_C[j] += sum_{i<j} w_pair[k] * (C[j] - C[i]) / (2 * dist_ij)
        
        # dists for pairs is pair_dists * 2
        pair_full_dists = pair_dists * 2.0
        # Avoid division by zero
        pair_full_dists[pair_full_dists < 1e-10] = 1e-10
        
        # Vectorized accumulation is tricky with np.add.at or similar
        # Loop over pairs might be slow if N is large, but N=50 is small.
        # 1225 pairs. A python loop over 1225 items is fine.
        
        for k in range(len(pair_dists)):
            i = i_idx[k]
            j = j_idx[k]
            d = pair_full_dists[k]
            vec = (C[i] - C[j]) / d
            grad_C[i] += w_pair[k] * vec
            grad_C[j] -= w_pair[k] * vec
            
        # 2. Boundary terms
        # w_bound = dw_dx[len(pair_dists):]
        w_bound = dw_dx[len(pair_dists):]
        # Reshape to (N, 3)
        w_bound = w_bound.reshape(N, 3)
        
        # d(0.5 - |C[i,a]|)/d(C[i,a]) = -sign(C[i,a])
        signs = np.sign(C)
        # Handle zeros
        signs[signs == 0] = 1.0
        
        grad_C -= w_bound * signs
        
        return obj, grad_C.flatten()

    for beta in betas:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(
                soft_min_obj_and_grad, 
                best_C.flatten(), 
                args=(beta,), 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            C_new = res.x.reshape(N, 3)
            
            # Check validity
            if not np.all(np.isfinite(C_new)):
                continue
                
            r_new = compute_radius(C_new)
            
            if r_new > best_radius:
                best_radius = r_new
                best_C = C_new
                
        except Exception as e:
            pass
            
    # Final check
    if time.time() > deadline:
         return best_C
         
    return best_C
