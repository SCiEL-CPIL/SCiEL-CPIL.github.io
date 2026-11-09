"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    
    if U_arr.shape != (n*m, rank):
        return False
    if V_arr.shape != (m*p, rank):
        return False
    if W_arr.shape != (p*n, rank):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_arr[i*m + j]
                v_val = V_arr[j*p + k]
                w_val = W_arr[k*n + i]
                s = np.dot(u_val, v_val * w_val)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_2x2x2(U, V, W, i_indices, j_indices, k_indices, col_offset):
    """
    Add Strassen decomposition for a 2x2x2 block to U, V, W.
    i_indices: [i0, i1]
    j_indices: [j0, j1]
    k_indices: [k0, k1]
    col_offset: starting column index in U, V, W
    """
    n, m, p = DIMENSIONS
    i0, i1 = i_indices
    j0, j1 = j_indices
    k0, k1 = k_indices
    
    # M1
    U[i0*m + j0, col_offset] += 1
    U[i1*m + j1, col_offset] += 1
    V[j0*p + k0, col_offset] += 1
    V[j1*p + k1, col_offset] += 1
    W[k0*n + i0, col_offset] += 1
    W[k1*n + i1, col_offset] += 1
    
    # M2
    U[i1*m + j0, col_offset + 1] += 1
    U[i1*m + j1, col_offset + 1] += 1
    V[j0*p + k0, col_offset + 1] += 1
    W[k0*n + i1, col_offset + 1] += 1
    W[k1*n + i1, col_offset + 1] += -1
    
    # M3
    U[i0*m + j0, col_offset + 2] += 1
    V[j0*p + k1, col_offset + 2] += 1
    V[j1*p + k1, col_offset + 2] += -1
    W[k1*n + i0, col_offset + 2] += 1
    
    # M4
    U[i1*m + j1, col_offset + 3] += 1
    V[j1*p + k0, col_offset + 3] += 1
    V[j0*p + k0, col_offset + 3] += -1
    W[k0*n + i1, col_offset + 3] += 1
    
    # M5
    U[i0*m + j0, col_offset + 4] += 1
    U[i0*m + j1, col_offset + 4] += 1
    V[j1*p + k1, col_offset + 4] += 1
    W[k1*n + i0, col_offset + 4] += 1
    
    # M6
    U[i1*m + j0, col_offset + 5] += 1
    U[i0*m + j0, col_offset + 5] += -1
    V[j0*p + k0, col_offset + 5] += 1
    V[j0*p + k1, col_offset + 5] += 1
    W[k1*n + i1, col_offset + 5] += 1
    
    # M7
    U[i0*m + j1, col_offset + 6] += 1
    U[i1*m + j1, col_offset + 6] += -1
    V[j1*p + k0, col_offset + 6] += 1
    V[j1*p + k1, col_offset + 6] += 1
    W[k0*n + i0, col_offset + 6] += 1

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Identify 2x2x2 blocks
    # i: 0-1, 2-3 (2 blocks)
    # j: 0-1, 2-3 (2 blocks, j=4 remains)
    # k: 0-1, 2-3, 4-5 (3 blocks)
    
    i_blocks = [(0, 1), (2, 3)]
    j_blocks = [(0, 1), (2, 3)]
    k_blocks = [(0, 1), (2, 3), (4, 5)]
    
    # Count total rank needed
    # Each 2x2x2 block uses 7 columns (Strassen)
    # Remaining terms (involving j=4) use 1 column each (schoolbook)
    
    num_strassen_blocks = len(i_blocks) * len(j_blocks) * len(k_blocks)
    strassen_cols = num_strassen_blocks * 7
    
    # Remaining terms: all (i, j=4, k) for i in range(n), k in range(p)
    remaining_terms = n * p  # 4 * 6 = 24
    remaining_cols = remaining_terms
    
    total_rank = strassen_cols + remaining_cols
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col_offset = 0
    
    # Add Strassen decompositions for each 2x2x2 block
    for i_idx in i_blocks:
        for j_idx in j_blocks:
            for k_idx in k_blocks:
                _strassen_2x2x2(U, V, W, i_idx, j_idx, k_idx, col_offset)
                col_offset += 7
    
    # Add schoolbook terms for j=4
    for i in range(n):
        for k in range(p):
            U[i*m + 4, col_offset] = 1
            V[4*p + k, col_offset] = 1
            W[k*n + i, col_offset] = 1
            col_offset += 1
    
    # Verify
    if not _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        # Fallback to schoolbook if Strassen doesn't work
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, c] = 1
                    V[j*p + k, c] = 1
                    W[k*n + i, c] = 1
                    c += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
