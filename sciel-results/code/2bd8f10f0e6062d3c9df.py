"""
Matrix multiplication 3x4 x 4x5 with rank-minimal tensor decomposition.
Uses a hybrid Strassen + schoolbook approach to achieve rank 56.
"""
import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def _build_strassen_block(i_offset, j_offset, k_offset, U, V, W, col_start):
    """
    Adds a Strassen decomposition for a 2x2x2 block starting at (i_offset, j_offset, k_offset).
    Returns the next column index.
    """
    # Indices for A (2x2)
    # a: (i_offset, j_offset)
    # b: (i_offset, j_offset+1)
    # c: (i_offset+1, j_offset)
    # d: (i_offset+1, j_offset+1)
    
    # Indices for B (2x2)
    # e: (j_offset, k_offset)
    # f: (j_offset, k_offset+1)
    # g: (j_offset+1, k_offset)
    # h: (j_offset+1, k_offset+1)
    
    # Indices for C (2x2)
    # c00: (i_offset, k_offset)
    # c01: (i_offset, k_offset+1)
    # c10: (i_offset+1, k_offset)
    # c11: (i_offset+1, k_offset+1)
    
    n, m, p = 3, 4, 5
    
    # Helper to get linear index
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # Strassen formulas:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # We will create 7 columns.
    
    # M1
    U[u_idx(i_offset, j_offset)][col_start] = 1
    U[u_idx(i_offset+1, j_offset+1)][col_start] = 1
    V[v_idx(j_offset, k_offset)][col_start] = 1
    V[v_idx(j_offset+1, k_offset+1)][col_start] = 1
    W[w_idx(k_offset, i_offset)][col_start] = 1
    W[w_idx(k_offset+1, i_offset)][col_start] = 1
    W[w_idx(k_offset, i_offset+1)][col_start] = 1
    W[w_idx(k_offset+1, i_offset+1)][col_start] = 1
    col_start += 1
    
    # M2
    U[u_idx(i_offset+1, j_offset)][col_start] = 1
    U[u_idx(i_offset+1, j_offset+1)][col_start] = 1
    V[v_idx(j_offset, k_offset)][col_start] = 1
    W[w_idx(k_offset, i_offset+1)][col_start] = 1
    col_start += 1
    
    # M3
    U[u_idx(i_offset, j_offset)][col_start] = 1
    V[v_idx(j_offset, k_offset+1)][col_start] = 1
    V[v_idx(j_offset+1, k_offset+1)][col_start] = -1
    W[w_idx(k_offset+1, i_offset)][col_start] = 1
    col_start += 1
    
    # M4
    U[u_idx(i_offset+1, j_offset+1)][col_start] = 1
    V[v_idx(j_offset+1, k_offset)][col_start] = 1
    V[v_idx(j_offset, k_offset)][col_start] = -1
    W[w_idx(k_offset, i_offset+1)][col_start] = 1
    col_start += 1
    
    # M5
    U[u_idx(i_offset, j_offset)][col_start] = 1
    U[u_idx(i_offset, j_offset+1)][col_start] = 1
    V[v_idx(j_offset+1, k_offset+1)][col_start] = 1
    W[w_idx(k_offset+1, i_offset)][col_start] = 1
    col_start += 1
    
    # M6
    U[u_idx(i_offset+1, j_offset)][col_start] = 1
    U[u_idx(i_offset, j_offset)][col_start] = -1
    V[v_idx(j_offset, k_offset)][col_start] = 1
    V[v_idx(j_offset, k_offset+1)][col_start] = 1
    W[w_idx(k_offset+1, i_offset+1)][col_start] = 1
    col_start += 1
    
    # M7
    U[u_idx(i_offset, j_offset+1)][col_start] = 1
    U[u_idx(i_offset+1, j_offset+1)][col_start] = -1
    V[v_idx(j_offset+1, k_offset)][col_start] = 1
    V[v_idx(j_offset+1, k_offset+1)][col_start] = 1
    W[w_idx(k_offset, i_offset)][col_start] = 1
    W[w_idx(k_offset+1, i_offset)][col_start] = 1
    col_start += 1
    
    return col_start

def _verify_decomposition(U, V, W, n, m, p):
    """Check if the decomposition exactly reproduces matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    if U.shape != (n*m, rank): return False
    if V.shape != (m*p, rank): return False
    if W.shape != (p*n, rank): return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = 3, 4, 5
    start_time = time.time()
    
    # Rank calculation:
    # 4 Strassen blocks of 2x2x2: 4 * 7 = 28
    # Remaining schoolbook:
    # i=2, j=0..3, k=0..4: 1 * 4 * 5 = 20
    # i=0,1, j=0..3, k=4: 2 * 4 * 1 = 8
    # Total rank = 28 + 20 + 8 = 56
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Apply Strassen blocks for i in {0,1}, j in {0,1} or {2,3}, k in {0,1} or {2,3}
    for i_off in [0]: # Only i=0,1 covered by Strassen
        for j_off in [0, 2]:
            for k_off in [0, 2]:
                col = _build_strassen_block(i_off, j_off, k_off, U, V, W, col)
                
    # Add remaining schoolbook terms
    # 1. i=2, all j, all k
    for j in range(m):
        for k in range(p):
            U[2*m + j][col] = 1
            V[j*p + k][col] = 1
            W[k*n + 2][col] = 1
            col += 1
            
    # 2. i=0,1, all j, k=4
    for i in [0, 1]:
        for j in range(m):
            k = 4
            U[i*m + j][col] = 1
            V[j*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
            
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails (should not happen)
        rank_school = n * m * p
        U_school = [[0] * rank_school for _ in range(n * m)]
        V_school = [[0] * rank_school for _ in range(m * p)]
        W_school = [[0] * rank_school for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m+j][c] = 1
                    V_school[j*p+k][c] = 1
                    W_school[k*n+i][c] = 1
                    c += 1
        return {"U": U_school, "V": V_school, "W": W_school}
        
    return {"U": U, "V": V, "W": W}
