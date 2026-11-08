"""Tensor decomposition for 2x5 x 5x6 matrix multiplication."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """Return rank-minimal tensor decomposition for 2x5 x 5x6 matmul.
    
    Uses a search strategy to find a lower rank decomposition if possible,
    starting from the known rank-54 solution.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 2, 5, 6
    
    # Target tensor: all ones
    # Shape: (n, m, p)
    # But we work with flattened indices for U, V, W
    # U: (n*m, K), V: (m*p, K), W: (p*n, K)
    
    # Construct the target tensor as a 3D array for easy evaluation
    T_target = np.ones((n, m, p))
    
    def compute_tensor(U, V, W, K):
        """Compute the tensor from factors U, V, W."""
        # U: (n*m, K), V: (m*p, K), W: (p*n, K)
        # Reshape U to (n, m, K), V to (m, p, K), W to (p, n, K)
        U_reshaped = U.reshape(n, m, K)
        V_reshaped = V.reshape(m, p, K)
        W_reshaped = W.reshape(p, n, K)
        
        # Sum over K: T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
        # Use einsum
        T = np.einsum('ijk,klm,ni->nil', U_reshaped, V_reshaped, W_reshaped)
        # Note: W is indexed by k, n (which is i). 
        # W_reshaped[k, i, r]
        # So T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
        # The einsum above:
        # U: i, j, r
        # V: j, k, r
        # W: k, i, r
        # Result: i, j, k
        # 'ijk, jkr, kir -> ijk'
        # Let's re-verify indices.
        # U[i*m + j, r] -> U_reshaped[i, j, r]
        # V[j*p + k, r] -> V_reshaped[j, k, r]
        # W[k*n + i, r] -> W_reshaped[k, i, r]
        # T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
        # np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        return T

    def objective(params, K):
        """Objective function for optimization."""
        # params is flattened U, V, W
        size_U = n * m * K
        size_V = m * p * K
        size_W = p * n * K
        
        U = params[0:size_U].reshape(n * m, K)
        V = params[size_U:size_U + size_V].reshape(m * p, K)
        W = params[size_U + size_V:].reshape(p * n, K)
        
        T = compute_tensor(U, V, W, K)
        error = np.sum((T - T_target) ** 2)
        return error

    # Initialize with the current rank-54 solution
    # We need to reconstruct the U, V, W from the current code logic
    # to have a good starting point.
    
    def get_initial_solution():
        rank = 54
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        j_pairs = [(0, 1), (2, 3)]
        k_pairs = [(0, 1), (2, 3), (4, 5)]
        
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                # 1
                U[0 * m + j0][col] += 1
                U[1 * m + j1][col] += 1
                V[j0 * p + k0][col] += 1
                V[j1 * p + k1][col] += 1
                W[k0 * n + 0][col] += 1
                W[k1 * n + 1][col] += 1
                col += 1
                # 2
                U[1 * m + j0][col] += 1
                U[1 * m + j1][col] += 1
                V[j0 * p + k0][col] += 1
                W[k0 * n + 1][col] += 1
                W[k1 * n + 1][col] -= 1
                col += 1
                # 3
                U[0 * m + j0][col] += 1
                V[j0 * p + k1][col] += 1
                V[j1 * p + k1][col] -= 1
                W[k1 * n + 0][col] += 1
                W[k1 * n + 1][col] += 1
                col += 1
                # 4
                U[1 * m + j1][col] += 1
                V[j1 * p + k0][col] += 1
                V[j0 * p + k0][col] -= 1
                W[k0 * n + 0][col] += 1
                W[k0 * n + 1][col] += 1
                col += 1
                # 5
                U[0 * m + j0][col] += 1
                U[0 * m + j1][col] += 1
                V[j1 * p + k1][col] += 1
                W[k0 * n + 0][col] -= 1
                W[k1 * n + 0][col] += 1
                col += 1
                # 6
                U[1 * m + j0][col] += 1
                U[0 * m + j0][col] -= 1
                V[j0 * p + k0][col] += 1
                V[j0 * p + k1][col] += 1
                W[k1 * n + 1][col] += 1
                col += 1
                # 7
                U[0 * m + j1][col] += 1
                U[1 * m + j1][col] -= 1
                V[j1 * p + k0][col] += 1
                V[j1 * p + k1][col] += 1
                W[k0 * n + 0][col] += 1
                col += 1
                
        for k in range(p):
            U[0 * m + 4][col] += 1
            V[4 * p + k][col] += 1
            W[k * n + 0][col] += 1
            col += 1
            U[1 * m + 4][col] += 1
            V[4 * p + k][col] += 1
            W[k * n + 1][col] += 1
            col += 1
            
        return U, V, W

    U_init, V_init, W_init = get_initial_solution()
    
    best_U, best_V, best_W = U_init, V_init, W_init
    best_rank = 54
    
    # Try to reduce rank
    for K in range(53, 40, -1):
        if time.time() > deadline - 10:
            break
            
        # Initialize params for K
        # Strategy: Take the first K terms from the initial solution
        # This is a heuristic, but might work if the dropped terms are small/linearly dependent
        # However, the terms are not ordered by importance.
        # Better: Random initialization?
        # Let's try random initialization with multiple restarts
        
        found = False
        for restart in range(5):
            if time.time() > deadline - 10:
                break
                
            # Random init
            U_rand = np.random.randn(n * m, K) * 0.1
            V_rand = np.random.randn(m * p, K) * 0.1
            W_rand = np.random.randn(p * n, K) * 0.1
            
            params = np.concatenate([U_rand.flatten(), V_rand.flatten(), W_rand.flatten()])
            
            # Optimize
            try:
                res = minimize(objective, params, args=(K,), method='L-BFGS-B', options={'maxiter': 50, 'maxfun': 100})
                if res.fun < 1e-6:
                    U_opt = res.x[0:n*m*K].reshape(n*m, K)
                    V_opt = res.x[n*m*K:n*m*K+m*p*K].reshape(m*p, K)
                    W_opt = res.x[n*m*K+m*p*K:].reshape(p*n, K)
                    best_U, best_V, best_W = U_opt, V_opt, W_opt
                    best_rank = K
                    found = True
                    break
            except Exception:
                pass
        
        if found:
            # Continue to try lower rank
            pass
        else:
            # If we couldn't find K, maybe K-1 is possible? 
            # But usually if K fails, lower ranks are harder.
            # However, sometimes random init for K fails but K-1 succeeds? Unlikely.
            # We will break to save time if we fail on the first attempt for a K?
            # No, we have 5 restarts.
            pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
