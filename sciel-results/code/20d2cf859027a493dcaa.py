"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

import math

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    # Target rank. We start with a lower bound guess.
    # Schoolbook is 84.
    # Let's try to find a decomposition with rank < 84.
    # We will use numerical optimization to find U, V, W.
    
    # To ensure speed and reliability, we will try a few ranks.
    # The lower bound is at least ceil(n*m*p / something)? No, rank >= max(n*m, m*p, p*n)? No.
    # Rank >= max(n*m, m*p, p*n) is not true.
    # Rank >= n*m*p / (max dim)? No.
    
    # Let's try rank 78, 79, 80, 81, 82, 83.
    # If we fail, return 84.
    
    best_rank = n * m * p  # 84
    best_solution = None
    
    # Pre-compute index mappings
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    def error_func(params, R):
        # params is a flat array of size (n*m + m*p + p*n) * R
        U = params[:n*m*R].reshape(n*m, R)
        V = params[n*m*R:(n*m+m*p)*R].reshape(m*p, R)
        W = params[(n*m+m*p)*R:].reshape(p*n, R)
        
        # Compute the tensor reconstruction
        # We need sum_r W[k,i,r] U[i,j1,r] V[j2,k,r] == delta(j1,j2)
        
        # Efficiently compute using einsum
        # U is (n*m, R) -> reshape to (n, m, R)
        U_reshaped = U.reshape(n, m, R)
        V_reshaped = V.reshape(m, p, R)
        W_reshaped = W.reshape(p, n, R)
        
        # We want to compute T[i, j1, j2, k] = sum_r W[k, i, r] U[i, j1, r] V[j2, k, r]
        # Note: V index is j*p+k. So V[j2, k, r].
        # W index is k*n+i. So W[k, i, r].
        
        # Let's compute the product U[i, j1, r] * V[j2, k, r] * W[k, i, r]
        # Sum over r.
        
        # This is a bit complex with einsum directly.
        # Let's compute M[i, j1, j2, k]
        
        # First, compute X[i, j1, j2, k] = sum_r U[i, j1, r] V[j2, k, r]
        # X = np.einsum('ijr,jkr->ijjk', U_reshaped, V_reshaped) -- wait, dimensions don't match.
        # U: (n, m, R)
        # V: (m, p, R)
        # We want to sum over r.
        # U[i, j1, r] * V[j2, k, r]
        # This gives a tensor of shape (n, m, m, p)
        X = np.einsum('ijr,jkr->ijjk', U_reshaped, V_reshaped)
        
        # Now multiply by W[k, i, r]? No, W is already summed over r in X?
        # No, X is sum_r U[i,j1,r] V[j2,k,r].
        # But we need sum_r W[k,i,r] U[i,j1,r] V[j2,k,r].
        # This is NOT the same as (sum_r U V) * W.
        # We need to keep r separate until the final sum.
        
        # Let's use a different approach.
        # Flatten r into the last dimension.
        # T[i, j1, j2, k] = sum_r W_reshaped[k, i, r] * U_reshaped[i, j1, r] * V_reshaped[j2, k, r]
        
        # We can do this in steps.
        # Step 1: Y[i, j1, j2, k, r] = U[i, j1, r] * V[j2, k, r]
        # This is large: 3*4*4*7*R. If R=80, 3*4*4*7*80 = 26880. Manageable.
        
        # Actually, we can just loop over r if R is small, or use broadcasting.
        # Let's use broadcasting.
        # U: (n, m, R) -> (n, m, 1, 1, R)
        # V: (m, p, R) -> (1, 1, m, p, R)
        # W: (p, n, R) -> (p, n, 1, 1, R)
        
        # We want result[i, j1, j2, k]
        # Term = U[i, j1, r] * V[j2, k, r] * W[k, i, r]
        
        # Let's compute P[i, j1, j2, k, r]
        U_b = U_reshaped[:, :, np.newaxis, np.newaxis, :]  # n, m, 1, 1, R
        V_b = V_reshaped[np.newaxis, np.newaxis, :, :, :] # 1, 1, m, p, R
        W_b = W_reshaped[:, :, np.newaxis, np.newaxis, :] # p, n, 1, 1, R
        
        # We need to align indices.
        # i is from U and W.
        # j1 is from U.
        # j2 is from V.
        # k is from V and W.
        
        # U[i, j1, r]
        # V[j2, k, r]
        # W[k, i, r]
        
        # Product:
        # P[i, j1, j2, k, r] = U[i, j1, r] * V[j2, k, r] * W[k, i, r]
        
        # This requires careful broadcasting.
        # U: (n, m, R) -> (n, m, 1, 1, R)
        # V: (m, p, R) -> (1, 1, m, p, R)
        # W: (p, n, R) -> (p, n, 1, 1, R)
        
        # We want to sum over r.
        # Result shape: (n, m, m, p)
        
        # Let's compute step by step.
        # Z[i, j1, j2, k, r] = U[i, j1, r] * V[j2, k, r]
        # U: (n, m, R)
        # V: (m, p, R)
        # We need to broadcast U and V.
        # U[i, j1, r]
        # V[j2, k, r]
        # Result Z[i, j1, j2, k, r]
        
        Z = U_reshaped[:, :, np.newaxis, np.newaxis, :] * V_reshaped[np.newaxis, np.newaxis, :, :, :]
        # Z shape: (n, m, m, p, R)
        
        # Now multiply by W[k, i, r]
        # W_reshaped is (p, n, R)
        # We need W[k, i, r] to match Z[i, j1, j2, k, r]
        # W index: k (0th), i (1st), r (2nd)
        # Z index: i (0th), j1 (1st), j2 (2nd), k (3rd), r (4th)
        
        # We need to transpose W to (n, 1, 1, p, R) to match i and k.
        # W_reshaped[k, i, r] -> W_transposed[i, k, r]
        W_trans = W_reshaped.T  # (n, p, R)
        W_b_trans = W_trans[:, np.newaxis, np.newaxis, :, :]  # (n, 1, 1, p, R)
        
        P = Z * W_b_trans
        # P shape: (n, m, m, p, R)
        
        T = np.sum(P, axis=4)
        # T shape: (n, m, m, p)
        
        # Target: T[i, j1, j2, k] should be 1 if j1==j2, else 0.
        # Note: i and k are free.
        
        # Create target tensor
        # Target[i, j1, j2, k] = delta(j1, j2)
        target = np.zeros((n, m, m, p))
        for j in range(m):
            target[:, j, j, :] = 1.0
        
        diff = T - target
        return np.sum(diff * diff)

    # Try different ranks
    for R in range(78, 85):
        np.random.seed(42)
        # Initialize with small random values
        U0 = np.random.randn(n*m, R) * 0.1
        V0 = np.random.randn(m*p, R) * 0.1
        W0 = np.random.randn(p*n, R) * 0.1
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        # Use L-BFGS-B
        try:
            res = minimize(error_func, x0, args=(R,), method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
            if res.success and res.fun < 1e-8:
                # Found a solution. Now round to integers/half-integers.
                U = res.x[:n*m*R].reshape(n*m, R)
                V = res.x[n*m*R:(n*m+m*p)*R].reshape(m*p, R)
                W = res.x[(n*m+m*p)*R:].reshape(p*n, R)
                
                # Round to nearest 0.5
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                
                # Verify exactness
                U_reshaped = U_int.reshape(n, m, R)
                V_reshaped = V_int.reshape(m, p, R)
                W_reshaped = W_int.reshape(p, n, R)
                
                Z = U_reshaped[:, :, np.newaxis, np.newaxis, :] * V_reshaped[np.newaxis, np.newaxis, :, :, :]
                W_trans = W_reshaped.T
                W_b_trans = W_trans[:, np.newaxis, np.newaxis, :, :]
                P = Z * W_b_trans
                T = np.sum(P, axis=4)
                
                target = np.zeros((n, m, m, p))
                for j in range(m):
                    target[:, j, j, :] = 1.0
                
                if np.allclose(T, target, atol=1e-9):
                    best_rank = R
                    best_solution = (U_int, V_int, W_int)
                    break
        except Exception as e:
            pass

    if best_solution:
        U, V, W = best_solution
        # Convert to lists of lists
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to schoolbook
        U = [[0] * (n * m * p) for _ in range(n * m)]
        V = [[0] * (n * m * p) for _ in range(m * p)]
        W = [[0] * (n * m * p) for _ in range(p * n)]
        idx = 0
        for j in range(m):
            for i in range(n):
                for k in range(p):
                    U[i * m + j][idx] = 1
                    V[j * p + k][idx] = 1
                    W[k * n + i][idx] = 1
                    idx += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
