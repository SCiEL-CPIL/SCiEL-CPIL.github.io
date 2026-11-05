import numpy as np
import time
import sys
from itertools import product
import scipy.optimize as opt

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 72
    
    # Target Tensor T[i, j, k] = 1
    # We represent T as a 3D array for easy einsum verification
    T_target = np.ones((n, m, p))
    
    # Function to check if U, V, W (reshaped to 3D) exactly decompose T_target
    # U: (n, m, r), V: (m, p, r), W: (p, n, r)
    def check_exact(U_3d, V_3d, W_3d, atol=1e-9):
        T_rec = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        return np.allclose(T_rec, T_target, atol=atol)

    # Function to generate schoolbook decomposition
    def get_schoolbook():
        r = rank_schoolbook
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
                    c += 1
        return U, V, W

    # Initial best solution (schoolbook)
    U_best, V_best, W_best = get_schoolbook()
    rank_best = rank_schoolbook

    # Candidate ranks to try, in increasing order of difficulty (higher rank is easier)
    # We want to find the MINIMAL rank, so we should try lower ranks first?
    # No, lower ranks are harder to find. Higher ranks (close to 72) are easier.
    # But we want the BEST score (lowest rank).
    # Strategy: Try a few promising lower ranks. If they fail, fallback to schoolbook.
    # Known bounds: rank >= 24.
    # Let's try ranks: 45, 50, 55, 60, 65.
    # If we find rank 55, score is 1/55. If we find rank 45, score is 1/45.
    # We should try to find the lowest possible rank.
    # However, optimization is stochastic. We might miss a low rank.
    # A safer bet is to try ranks that are known to be achievable or likely.
    # For 3x4x6, let's try a range.
    
    candidate_ranks = [40, 45, 50, 55, 60]
    
    def optimize_rank(r, max_restarts=5, max_iter=100):
        if time.time() > deadline - 5.0:
            return None
            
        size_U = n * m * r
        size_V = m * p * r
        size_W = p * n * r
        total_size = size_U + size_V + size_W
        
        def objective_and_grad(x):
            U_flat = x[:size_U]
            V_flat = x[size_U:size_U+size_V]
            W_flat = x[size_U+size_V:]
            
            U = U_flat.reshape(n, m, r)
            V = V_flat.reshape(m, p, r)
            W = W_flat.reshape(p, n, r)
            
            T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
            err = T_rec - T_target
            loss = np.sum(err**2)
            
            # Gradients
            # dL/dU[i,j,c] = 2 * sum_k err[i,j,k] * V[j,k,c] * W[k,i,c]
            grad_U = 2 * np.einsum('ijk,jkc,kic->ijc', err, V, W)
            grad_V = 2 * np.einsum('ijk,ijc,kic->jkc', err, U, W)
            grad_W = 2 * np.einsum('ijk,ijc,jkc->kic', err, U, V)
            
            grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            return loss, grad

        for restart in range(max_restarts):
            if time.time() > deadline - 2.0:
                break
                
            np.random.seed(restart * 100 + r) # Deterministic restarts for reproducibility
            x0 = np.random.randn(total_size) * 0.5
            
            res = opt.minimize(
                objective_and_grad, 
                x0, 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 20}
            )
            
            if res.fun < 1e-6: # Good enough to try rounding
                U_flat = res.x[:size_U]
                V_flat = res.x[size_U:size_U+size_V]
                W_flat = res.x[size_U+size_V:]
                
                U = U_flat.reshape(n, m, r)
                V = V_flat.reshape(m, p, r)
                W = W_flat.reshape(p, n, r)
                
                # Round to nearest 0.5
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                if check_exact(U_rounded, V_rounded, W_rounded):
                    # Convert to flat matrices for return
                    U_ret = U_rounded.flatten(axis=1) # (n*m, r)
                    V_ret = V_rounded.flatten(axis=1) # (m*p, r)
                    W_ret = W_rounded.flatten(axis=1) # (p*n, r)
                    return U_ret, V_ret, W_ret
                    
        return None

    # Try candidate ranks
    # We want the lowest rank, so we iterate from low to high?
    # But low ranks are harder. If we spend too much time on rank 40 and fail, we might not have time for 50.
    # Let's allocate time proportionally or just try a few.
    # Let's try 50, 55, 60 first as they are more likely to succeed.
    # If we find rank 50, we can stop? No, we want the MINIMAL rank.
    # But finding rank 45 is much harder than 55.
    # Given the time constraint, it's safer to find a valid decomposition of rank ~50-60 than to risk failing all and returning 72.
    # However, if we find 50, we should check if 45 is possible?
    # Let's just try a few and keep the best (lowest rank) found.
    
    results = {}
    for r in [50, 55, 60, 45, 40]: # Order: likely to succeed first
        if time.time() > deadline - 10.0:
            break
            
        res = optimize_rank(r, max_restarts=3, max_iter=50)
        if res is not None:
            U_r, V_r, W_r = res
            results[r] = (U_r, V_r, W_r)
            # If we found a rank, we can break if we are confident it's minimal?
            # No, we don't know if 45 is possible. But 45 is very hard.
            # Let's keep searching for lower ranks if time permits.
            # But we should prioritize finding ANY improvement over 72.
            
    # Select the best (lowest rank) result
    if results:
        min_rank = min(results.keys())
        U_best, V_best, W_best = results[min_rank]
        rank_best = min_rank
        
    # If no improvement found, use schoolbook
    # U_best, V_best, W_best are already set to schoolbook initially
    
    # Convert to lists
    return {
        "U": U_best.tolist(),
        "V": V_best.tolist(),
        "W": W_best.tolist()
    }
# EVOLVE-BLOCK-END
