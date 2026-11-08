import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    import time
    _t0 = time.time()
    _deadline = 140.0
    
    n, m, p = DIMENSIONS
    
    block_size = 2
    blocks_per_dim = n // block_size # 3
    
    rank = (blocks_per_dim ** 3) * 7
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    strassen_patterns = [
        # M0 = (A00+A11)(B00+B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M1 = (A10+A11)B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M2 = A00(B01-B11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M3 = A11(B10-B00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M4 = (A00+A01)B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M5 = (A10-A00)(B00+B01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M6 = (A01-A11)(B10+B11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    col_idx = 0
    for I in range(blocks_per_dim):
        for J in range(blocks_per_dim):
            for K in range(blocks_per_dim):
                for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                    if time.time() > _deadline:
                        return {"U": U, "V": V, "W": W}
                    
                    # Update U
                    for (pi, pj), val in u_coeffs.items():
                        r = I * block_size + pi
                        c = J * block_size + pj
                        U[r * m + c, col_idx] += val
                    
                    # Update V
                    for (pi, pj), val in v_coeffs.items():
                        r = J * block_size + pi
                        c = K * block_size + pj
                        V[r * p + c, col_idx] += val
                    
                    # Update W
                    for (pi, pj), val in w_coeffs.items():
                        k = K * block_size + pj
                        i = I * block_size + pi
                        W[k * n + i, col_idx] += val
                    
                    col_idx += 1
                    
    # Verification
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should be 1 for all i,j,k
    # We can check a few points or all points.
    # 6*6*6 = 216 points. 189 terms. 216*189 ~ 40k ops. Very fast.
    
    # Construct full tensor check
    # T_expected = ones((n, m, p))
    # T_calc = np.einsum('ir,jr,kr->ijk', U.reshape(n,m,rank), V.reshape(m,p,rank), W.reshape(p,n,rank))
    # Note: W index is k*n+i. So W.reshape(p, n, rank) gives W[k, i, r].
    # U index i*m+j. U.reshape(n, m, rank) gives U[i, j, r].
    # V index j*p+k. V.reshape(m, p, rank) gives V[j, k, r].
    
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # Compute T_calc[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
    # Using einsum
    T_calc = np.einsum('ijr,jkr,kr->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    # Check if T_calc is all ones
    if not np.all(T_calc == 1):
        # Fallback to trivial decomposition if Strassen fails
        # Rank = n*m*p
        rank_trivial = n * m * p
        U_t = np.zeros((n * m, rank_trivial), dtype=int)
        V_t = np.zeros((m * p, rank_trivial), dtype=int)
        W_t = np.zeros((p * n, rank_trivial), dtype=int)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U_t[i * m + j, r] = 1
                    V_t[j * p + k, r] = 1
                    W_t[k * n + i, r] = 1
        
        return {"U": U_t, "V": V_t, "W": W_t}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
