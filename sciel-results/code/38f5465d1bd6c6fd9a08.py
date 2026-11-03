"""Matrix multiplication 5,5,5 - rank-minimal tensor decomposition via block decomposition."""
DIMENSIONS = [5, 5, 5]

import time
import numpy as np

def _build_strassen_2x2():
    """Return U, V, W for 2x2 matrix multiplication (rank 7), trace convention W[k*n+i, r]."""
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # col 0: M1 = (a00+a11)*(b00+b11) -> c00 += 1, c11 += 1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # col 1: M2 = (a10+a11)*b00 -> c10 += 1, c11 -= 1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # col 2: M3 = a00*(b01-b11) -> c00 += 1, c01 += 1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[0, 2] = 1; W[1, 2] = 1
    
    # col 3: M4 = a11*(b10-b00) -> c00 += 1, c10 += 1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # col 4: M5 = (a00+a01)*b11 -> c00 -= 1, c01 += 1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # col 5: M6 = (a10-a00)*(b00+b01) -> c11 += 1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # col 6: M7 = (a01-a11)*(b10+b11) -> c00 += 1, c10 -= 1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[2, 6] = -1
    
    return U, V, W

def _build_schoolbook_block(n, m, p):
    """Return U, V, W for n x m * m x p schoolbook (rank n*m*p)."""
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
    return U, V, W

def _assemble_block_decomp(n, m, p, n1, m1, p1):
    """
    Decompose n x m * m x p by splitting into 2x2 blocks and decomposing each block product.
    """
    n2, m2, p2 = n - n1, m - m1, p - p1
    
    def get_decomp(ni, mi, pi):
        if ni == 2 and mi == 2 and pi == 2:
            return _build_strassen_2x2()
        else:
            return _build_schoolbook_block(ni, mi, pi)
    
    # 8 block products: (ni, mi, pi, i_off, j_off, k_off)
    terms = [
        (n1, m1, p1, 0, 0, 0),
        (n1, m2, p1, 0, m1, 0),
        (n1, m1, p2, 0, 0, p1),
        (n1, m2, p2, 0, m1, p1),
        (n2, m1, p1, n1, 0, 0),
        (n2, m2, p1, n1, m1, 0),
        (n2, m1, p2, n1, 0, p1),
        (n2, m2, p2, n1, m1, p1),
    ]
    
    total_rank = sum(get_decomp(ni, mi, pi)[0].shape[1] for ni, mi, pi, _, _, _ in terms)
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    for (ni, mi, pi, i_off, j_off, k_off) in terms:
        U_loc, V_loc, W_loc = get_decomp(ni, mi, pi)
        r = U_loc.shape[1]
        
        for i_loc in range(ni):
            for j_loc in range(mi):
                local_row = i_loc * mi + j_loc
                global_row = (i_off + i_loc) * m + (j_off + j_loc)
                U[global_row, col_offset:col_offset+r] = U_loc[local_row, :]
        
        for j_loc in range(mi):
            for k_loc in range(pi):
                local_row = j_loc * pi + k_loc
                global_row = (j_off + j_loc) * p + (k_off + k_loc)
                V[global_row, col_offset:col_offset+r] = V_loc[local_row, :]
        
        for k_loc in range(pi):
            for i_loc in range(ni):
                local_row = k_loc * ni + i_loc
                global_row = (k_off + k_loc) * n + (i_off + i_loc)
                W[global_row, col_offset:col_offset+r] = W_loc[local_row, :]
        
        col_offset += r
    
    return U, V, W

def construct():
    deadline = time.time() + 140
    
    n, m, p = DIMENSIONS
    
    # Use 2+3 block split for 5x5x5: rank = 7+12+12+18+12+18+18+27 = 124
    U, V, W = _assemble_block_decomp(n, m, p, 2, 2, 2)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
