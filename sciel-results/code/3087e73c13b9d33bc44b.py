import numpy as np
import time
import itertools
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110  # Safety margin
    
    # Schoolbook solution as fallback
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_result = None
    best_rank = n * m * p

    # Target tensor: all ones
    # T[i, j, k] = 1
    # Equation: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
    
    # Reshape helper
    def reshape_factors(U, V, W, R):
        # U: (n*m, R) -> (n, m, R)
        # V: (m*p, R) -> (m, p, R)
        # W: (p*n, R) -> (p, n, R)
        U3 = U.reshape(n, m, R)
        V3 = V.reshape(m, p, R)
        W3 = W.reshape(p, n, R)
        return U3, V3, W3

    def compute_error(params, R):
        U, V, W = params
        U3, V3, W3 = reshape_factors(U, V, W, R)
        # Compute T[i,j,k] = sum_r U3[i,j,r] V3[j,k,r] W3[k,i,r]
        # Using einsum
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        target = np.ones((n, m, p))
        error = T - target
        return error.flatten()

    # Try ranks from lower bound up to schoolbook rank
    # Lower bound is roughly max(n*m, m*p, p*n) = 24
    # Let's start from 30 to be safe and efficient
    for R in range(30, best_rank + 1):
        if time.time() > deadline:
            break
            
        found = False
        # Number of restarts depends on time
        # If R is close to 72, it's easier, so fewer restarts needed?
        # Actually, lower R is harder, so more restarts might be needed, but we have less time.
        # Let's do 5 restarts for each R.
        for restart in range(5):
            if time.time() > deadline:
                break
                
            # Random initialization
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # Scale initialization to help convergence
            # The product of three random normal variables has small mean but variance grows.
            # We want sum of products to be 1.
            # If R is large, each term should be small.
            # std of product of 3 normals is 1.
            # Sum of R such terms has std sqrt(R).
            # We want sum to be 1, so scale by 1/sqrt(R)?
            # Or just let the optimizer handle it.
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = least_squares(compute_error, params, args=(R,), method='lm', max_nfev=100)
            except Exception:
                continue
                
            if res.cost < 1e-8:
                # Found a solution
                U_sol = res.x[:n*m*R].reshape(n*m, R)
                V_sol = res.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                W_sol = res.x[n*m*R + m*p*R:].reshape(p*n, R)
                
                # Snap to half-integers
                U_snap = np.round(U_sol * 2) / 2
                V_snap = np.round(V_sol * 2) / 2
                W_snap = np.round(W_sol * 2) / 2
                
                # Verify exactness
                U3s, V3s, W3s = reshape_factors(U_snap, V_snap, W_snap, R)
                T_snap = np.einsum('ijr,jkr,kir->ijk', U3s, V3s, W3s)
                if np.allclose(T_snap, np.ones((n, m, p)), atol=1e-10):
                    best_rank = R
                    best_result = {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
                    found = True
                    break
        
        if found:
            break

    if best_result is not None:
        return best_result
    else:
        return get_schoolbook()
# EVOLVE-BLOCK-END
