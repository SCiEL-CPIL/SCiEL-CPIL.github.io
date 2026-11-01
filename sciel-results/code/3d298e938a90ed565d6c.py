import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    # Target Tensor T
    # T[i, j, k] = 1
    # Flattened index: (i*m + j)*p + k
    T_target = np.ones(n * m * p)

    def compute_residual(params):
        # params shape: (n*m*rank + m*p*rank + p*n*rank,)
        rank = 20
        U_size = n * m * rank
        V_size = m * p * rank
        W_size = p * n * rank
        
        U_flat = params[:U_size]
        V_flat = params[U_size:U_size + V_size]
        W_flat = params[U_size + V_size:U_size + V_size + W_size]
        
        U = U_flat.reshape(n * m, rank)
        V = V_flat.reshape(m * p, rank)
        W = W_flat.reshape(p * n, rank)
        
        # Compute T_computed
        # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Vectorized computation
        # Expand dimensions for broadcasting
        # U: (n*m, rank) -> (n, m, 1, rank)
        # V: (m*p, rank) -> (1, m, p, rank)
        # W: (p*n, rank) -> (p, 1, n, rank)
        
        U_reshaped = U.reshape(n, m, rank)
        V_reshaped = V.reshape(m, p, rank)
        W_reshaped = W.reshape(p, n, rank)
        
        # T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
        
        # Multiply U and V: (n, m, p, rank)
        UV = U_reshaped[:, :, np.newaxis, :] * V_reshaped[np.newaxis, :, :, :]
        # UV[i, j, k, r] = U[i, j, r] * V[j, k, r]
        
        # Multiply by W: (n, m, p)
        # W[k, i, r] needs to be aligned with UV[i, j, k, r]
        # W_reshaped is (p, n, rank). We need (n, p, rank) or similar.
        # Let's transpose W to (n, p, rank)
        W_transposed = W_reshaped.transpose(1, 0, 2) # (n, p, rank)
        
        # Now UV is (n, m, p, rank). W_transposed is (n, p, rank).
        # We need to sum over r and align indices.
        # T[i, j, k] = sum_r UV[i, j, k, r] * W_transposed[i, k, r]
        
        # UV: (n, m, p, rank)
        # W_trans: (n, 1, p, rank)
        W_expanded = W_transposed[:, np.newaxis, :, :]
        
        T_computed = np.sum(UV * W_expanded, axis=3)
        
        # T_computed shape: (n, m, p)
        # Target shape: (n*m*p,)
        # Flatten T_computed in order (i, j, k)
        T_flat = T_computed.reshape(n * m * p)
        
        return T_flat - T_target

    # Initial guess: Use Strassen rank-21, but truncate to 20?
    # Or random?
    # Random is safer to avoid local minima of the specific Strassen structure.
    
    rank = 20
    U_size = n * m * rank
    V_size = m * p * rank
    W_size = p * n * rank
    total_size = U_size + V_size + W_size
    
    best_result = None
    best_error = float('inf')
    
    # Try multiple random starts
    for attempt in range(10):
        if time.time() > deadline - 20:
            break
            
        # Random initialization
        np.random.seed(attempt)
        params0 = np.random.randn(total_size)
        
        try:
            res = least_squares(compute_residual, params0, method='lm', max_nfev=1000)
            error = res.cost
            if error < best_error:
                best_error = error
                best_params = res.x
                
                # If error is very small, try to recover integer/half-integer solution
                if error < 1e-8:
                    U_flat = best_params[:U_size]
                    V_flat = best_params[U_size:U_size + V_size]
                    W_flat = best_params[U_size + V_size:U_size + V_size + W_size]
                    
                    U = U_flat.reshape(n * m, rank)
                    V = V_flat.reshape(m * p, rank)
                    W = W_flat.reshape(p * n, rank)
                    
                    # Round to nearest 0.5
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    # Verify
                    U_r = U_rounded.reshape(n, m, rank)
                    V_r = V_rounded.reshape(m, p, rank)
                    W_r = W_rounded.reshape(p, n, rank)
                    W_r_t = W_r.transpose(1, 0, 2)
                    
                    UV_r = U_r[:, :, np.newaxis, :] * V_r[np.newaxis, :, :, :]
                    W_r_e = W_r_t[:, np.newaxis, :, :]
                    T_check = np.sum(UV_r * W_r_e, axis=3).reshape(n * m * p)
                    
                    if np.allclose(T_check, T_target, atol=1e-8):
                        return {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
        except Exception as e:
            pass

    # Fallback to Rank 21 Strassen
    rank21 = 21
    U21 = np.zeros((n * m, rank21))
    V21 = np.zeros((m * p, rank21))
    W21 = np.zeros((p * n, rank21))
    
    U_coeffs = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1}
    ]
    V_coeffs = [
        [(0, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(0, 1, 1), (1, 1, -1)],
        [(1, 0, 1), (0, 0, -1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (1, 1, 1)]
    ]
    W_map = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    for r in range(3):
        for k in range(7):
            col = r * 7 + k
            for idx, val in U_coeffs[k].items():
                U21[idx, col] = val
            for j_local, k_local, val in V_coeffs[k]:
                global_k = 2 * r + k_local
                v_idx = j_local * p + global_k
                V21[v_idx, col] = val
            for i, j_local, val in W_map[k]:
                global_k = 2 * r + j_local
                w_idx = global_k * n + i
                W21[w_idx, col] = val
                
    return {"U": U21.tolist(), "V": V21.tolist(), "W": W21.tolist()}
# EVOLVE-BLOCK-END
