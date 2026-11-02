"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Initial guess: Jittered grid
    # A 5x10 or 7x8 grid is better. 50 points.
    # Let's try a 5x10 grid stretched to fit.
    # Or just use the seed's grid and jitter it.
    m = int(np.ceil(np.sqrt(N)))
    # Create a grid of m x m
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    x0 = grid[:N].copy()
    
    # Add small random jitter to break symmetry
    np.random.seed(42)
    x0 += np.random.normal(0, 0.01, x0.shape)
    
    # Ensure initial points are inside
    x0 = np.clip(x0, -0.49, 0.49)
    
    best_x = x0
    best_score = -np.inf
    
    def calculate_score(x):
        # x is (N, 2)
        # Calculate pairwise distances
        dists = np.linalg.norm(x[:, np.newaxis, :] - x[np.newaxis, :, :], axis=-1)
        # Zero out diagonal
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Calculate wall distances
        # min_i min_k (0.5 - |x_ik|)
        # x_ik is coordinate. 0.5 - |coord|
        # min over all coords
        wall_dists = 0.5 - np.abs(x)
        min_wall_dist = np.min(wall_dists)
        
        # The radius is min(min_pair_dist / 2, min_wall_dist)
        r = min(min_pair_dist / 2.0, min_wall_dist)
        
        # Check validity
        if not np.all(np.isfinite(x)) or np.any(np.abs(x) > 0.5 + 1e-9):
            return -np.inf
        return r

    # Soft-min objective function
    # Minimize S(x) = 1/beta * log( sum exp(-beta * d_i) )
    # d_i are the relevant distances (pair/2 and wall)
    
    def objective(x_flat, beta):
        x = x_flat.reshape(N, 2)
        
        # Pairwise distances
        # Efficient calculation for N=50
        # diff = x[:, None, :] - x[None, :, :]
        # dists = np.linalg.norm(diff, axis=-1)
        # This creates a 50x50 matrix.
        
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        # We need min(dists/2, wall_dists)
        # Let's construct the array of all relevant distances
        # Pair distances: dists / 2
        pair_dists = dists / 2.0
        
        # Wall distances: 0.5 - |x_ik|
        wall_dists = 0.5 - np.abs(x)
        
        # Combine all distances into a single flat array
        # Exclude inf values from pair_dists
        valid_pairs = pair_dists[np.isfinite(pair_dists)]
        
        all_dists = np.concatenate([valid_pairs, wall_dists.ravel()])
        
        # Soft min: (1/beta) * log(sum(exp(-beta * d)))
        # To avoid overflow/underflow, subtract max? No, we want min.
        # exp(-beta * d) is largest for smallest d.
        # If d is very small, -beta*d is large positive? No, -beta * small = small negative.
        # exp(small negative) ~ 1.
        # If d is large, -beta*d is large negative. exp(large negative) ~ 0.
        # So sum is dominated by smallest d.
        
        # Numerical stability:
        # log(sum(exp(-beta * d))) = -beta * d_min + log(sum(exp(-beta * (d - d_min))))
        # We don't know d_min exactly, but we can use a stable logsumexp.
        # scipy.special.logsumexp is available.
        
        try:
            from scipy.special import logsumexp
            # We want log(sum(exp(-beta * d)))
            # This is logsumexp(-beta * all_dists)
            val = logsumexp(-beta * all_dists)
            return val / beta
        except Exception:
            return 1e10

    # Optimization loop with increasing beta
    betas = [10, 50, 200, 1000, 5000]
    
    current_x = x0
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
            
        # Define the objective for this beta
        def obj_wrapper(x_flat):
            return objective(x_flat, beta)
        
        # Run L-BFGS-B
        try:
            res = minimize(obj_wrapper, current_x.ravel(), method='L-BFGS-B', 
                           options={'maxiter': 1000, 'maxfun': 10000, 'ftol': 1e-10})
            
            if res.success or res.fun < objective(current_x.ravel(), beta):
                current_x = res.x.reshape(N, 2)
                
                # Calculate actual score
                score = calculate_score(current_x)
                if score > best_score:
                    best_score = score
                    best_x = current_x.copy()
                    
        except Exception as e:
            # If optimization fails, continue with current best
            pass

    # Final polish: Use a very high beta or just return best_x
    # Ensure best_x is valid
    if best_score == -np.inf:
        # Fallback to initial
        best_x = x0
        best_score = calculate_score(best_x)

    return best_x
# EVOLVE-BLOCK-END
