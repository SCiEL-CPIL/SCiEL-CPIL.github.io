"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

DIMENSIONS = [3, 5, 5]

def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from scipy.optimize import least_squares

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _tensor_error(U, V, W, n, m, p):
    """Compute the tensor reconstruction error."""
    rank = len(U[0]) if U else 0
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                err += abs(val - 1.0)
    return err

def _flatten(U, V, W):
    return np.concatenate([np.array(U).flatten(), np.array(V).flatten(), np.array(W).flatten()])

def _unflatten(x, n, m, p, rank):
    size_U = n * m * rank
    size_V = m * p * rank
    size_W = p * n * rank
    U = x[:size_U].reshape(n * m, rank)
    V = x[size_U:size_U + size_V].reshape(m * p, rank)
    W = x[size_U + size_V:].reshape(p * n, rank)
    return U, V, W

def _residuals(x, n, m, p, rank):
    U, V, W = _unflatten(x, n, m, p, rank)
    res = np.zeros(n * m * p)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :]) * W[k*n+i, :]
                # Wait, the term is sum_c U_ijk V_jkc W_kic.
                # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                # This is not a simple dot product of U and V multiplied by W element-wise?
                # Yes, for each c, we have a product.
                # sum_c (U_i_j_c * V_j_k_c) * W_k_i_c
                # Let M = U * V^T? No.
                # We can compute this as:
                # For each c, let T_c[i,j,k] = U[i,m+j,c] * V[j,p+k,c] * W[k,n+i,c]
                # We want sum_c T_c[i,j,k] = 1.
                
                # Efficient computation:
                # u_row = U[i*m+j, :]
                # v_row = V[j*p+k, :]
                # w_row = W[k*n+i, :]
                # val = np.sum(u_row * v_row * w_row)
                
                u_row = U[i*m+j, :]
                v_row = V[j*p+k, :]
                w_row = W[k*n+i, :]
                val = np.sum(u_row * v_row * w_row)
                res[idx] = val - 1.0
                idx += 1
    return res

def _round_to_half(x):
    return round(x * 2) / 2.0

def _try_reduce_rank(max_rank, time_budget):
    """Try to find a decomposition with rank < max_rank."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    best_result = _schoolbook()
    
    # We try to reduce rank by 1, 2, etc.
    # Limit search to a few ranks to save time
    for target_rank in range(max_rank - 1, max(max_rank - 5, 1)):
        if time.time() - start_time > time_budget - 10:
            break
            
        # Try multiple random initializations
        for attempt in range(5):
            if time.time() - start_time > time_budget - 10:
                break
                
            # Random initialization
            U_init = np.random.randn(n * m, target_rank) * 0.5
            V_init = np.random.randn(m * p, target_rank) * 0.5
            W_init = np.random.randn(p * n, target_rank) * 0.5
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = least_squares(_residuals, x0, args=(n, m, p, target_rank), method='trf', max_nfev=1000)
                
                if res.cost < 1e-10:
                    U, V, W = _unflatten(res.x, n, m, p, target_rank)
                    
                    # Round to nearest half-integer
                    U_rounded = np.vectorize(_round_to_half)(U)
                    V_rounded = np.vectorize(_round_to_half)(V)
                    W_rounded = np.vectorize(_round_to_half)(W)
                    
                    # Convert to lists
                    U_list = U_rounded.tolist()
                    V_list = V_rounded.tolist()
                    W_list = W_rounded.tolist()
                    
                    # Verify exactness
                    if _tensor_error(U_list, V_list, W_list, n, m, p) < 1e-9:
                        return {"U": U_list, "V": V_list, "W": W_list}
                        
            except Exception:
                pass

    return best_result

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    time_budget = 110  # Leave margin for 120s limit
    
    start_time = time.time()
    
    result = _try_reduce_rank(schoolbook_rank, time_budget)
    
    return result
# EVOLVE-BLOCK-END
