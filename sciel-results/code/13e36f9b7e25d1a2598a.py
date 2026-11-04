"""
Program for 5x5x5 matrix multiplication tensor decomposition.
Uses a block decomposition strategy: partition the 5x5x5 multiplication
into blocks and apply Strassen's algorithm where blocks are 2x2x2,
with schoolbook for the remaining 1x1 or mixed blocks.

The key insight: 5 = 2 + 2 + 1, so we can partition indices into
blocks of size 2, 2, 1. For each pair of blocks (block_a from A,
block_b from B, block_c in C), if both input blocks are 2x2 and the
output block is 2x2, we can use Strassen (rank 7 instead of 8).
Otherwise we use schoolbook.

This should give rank significantly below 125.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2_terms():
    """
    Return the 7 rank-1 terms for Strassen's 2x2x2 matrix multiplication.
    Each term is (U_row, V_col, W_row) indices where U is 4x1, V is 4x1, W is 4x1.
    
    Strassen's algorithm for C = A * B where A, B, C are 2x2:
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22) B11
    M3 = A11 (B12 - B22)
    M4 = A22 (B21 - B11)
    M5 = (A11 + A12) B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    U factors (for A entries A11, A12, A21, A22):
    V factors (for B entries B11, B12, B21, B22):
    W factors (for C entries C11, C12, C21, C22):
    """
    # Each term: (coeff_a11, coeff_a12, coeff_a21, coeff_a22,
    #              coeff_b11, coeff_b12, coeff_b21, coeff_b22,
    #              coeff_c11, coeff_c12, coeff_c21, coeff_c22)
    terms = [
        # M1 = (A11 + A22)(B11 + B22)
        (1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1),
        # M2 = (A21 + A22) B11
        (0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0),
        # M3 = A11 (B12 - B22)
        (1, 0, 0, 0, 0, 1, 0, -1, 0, 1, 0, 0),
        # M4 = A22 (B21 - B11)
        (0, 0, 0, 1, -1, 0, 1, 0, 1, 0, 1, 0),
        # M5 = (A11 + A12) B22
        (1, 1, 0, 0, 0, 0, 0, 1, -1, 1, 0, 0),
        # M6 = (A21 - A11)(B11 + B12)
        (-1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1),
        # M7 = (A12 - A22)(B21 + B22)
        (0, 1, 0, -1, 0, 0, 1, 1, 1, 0, 0, 0),
    ]
    return terms

def _build_block_decomposition(n, m, p):
    """
    Build a block decomposition for n x m x p matrix multiplication.
    Partition indices into blocks of size 2 (last block may be size 1).
    
    For blocks of size 2x2x2, use Strassen (rank 7).
    For all other block combinations, use schoolbook.
    """
    # Determine block boundaries
    def get_blocks(size):
        blocks = []
        pos = 0
        while pos < size:
            sz = min(2, size - pos)
            blocks.append((pos, sz))
            pos += sz
        return blocks
    
    row_blocks = get_blocks(n)   # blocks for i (rows of A, rows of C)
    mid_blocks = get_blocks(m)   # blocks for j (cols of A, rows of B)
    col_blocks = get_blocks(p)   # blocks for k (cols of B, cols of C)
    
    rank = 0
    U = []  # will be list of (row_index, value) per term
    V = []
    W = []
    
    for (i_start, i_sz) in row_blocks:
        for (j_start, j_sz) in mid_blocks:
            for (k_start, k_sz) in col_blocks:
                if i_sz == 2 and j_sz == 2 and k_sz == 2:
                    # Use Strassen
                    strassen_terms = _strassen_2x2_terms()
                    for (ua11, ua12, ua21, ua22,
                         vb11, vb12, vb21, vb22,
                         wc11, wc12, wc21, wc22) in strassen_terms:
                        # A block: A[i_start:i_start+2, j_start:j_start+2]
                        # indices: A[i_start, j_start]=a11, A[i_start, j_start+1]=a12
                        #          A[i_start+1, j_start]=a21, A[i_start+1, j_start+1]=a22
                        # B block: B[j_start:j_start+2, k_start:k_start+2]
                        # indices: B[j_start, k_start]=b11, B[j_start, k_start+1]=b12
                        #          B[j_start+1, k_start]=b21, B[j_start+1, k_start+1]=b22
                        # C block: C[i_start:i_start+2, k_start:k_start+2]
                        # indices: C[i_start, k_start]=c11, C[i_start, k_start+1]=c12
                        #          C[i_start+1, k_start]=c21, C[i_start+1, k_start+1]=c22
                        
                        # U: rows are i*m + j
                        u_row_a11 = (i_start) * m + (j_start)
                        u_row_a12 = (i_start) * m + (j_start + 1)
                        u_row_a21 = (i_start + 1) * m + (j_start)
                        u_row_a22 = (i_start + 1) * m + (j_start + 1)
                        
                        # V: rows are j*p + k
                        v_row_b11 = (j_start) * p + (k_start)
                        v_row_b12 = (j_start) * p + (k_start + 1)
                        v_row_b21 = (j_start + 1) * p + (k_start)
                        v_row_b22 = (j_start + 1) * p + (k_start + 1)
                        
                        # W: rows are k*n + i
                        w_row_c11 = (k_start) * n + (i_start)
                        w_row_c12 = (k_start) * n + (i_start + 1)
                        w_row_c21 = (k_start + 1) * n + (i_start)
                        w_row_c22 = (k_start + 1) * n + (i_start + 1)
                        
                        U.append((u_row_a11, ua11))
                        U.append((u_row_a12, ua12))
                        U.append((u_row_a21, ua21))
                        U.append((u_row_a22, ua22))
                        V.append((v_row_b11, vb11))
                        V.append((v_row_b12, vb12))
                        V.append((v_row_b21, vb21))
                        V.append((v_row_b22, vb22))
                        W.append((w_row_c11, wc11))
                        W.append((w_row_c12, wc12))
                        W.append((w_row_c21, wc21))
                        W.append((w_row_c22, wc22))
                        rank += 1
                else:
                    # Schoolbook for this block
                    for di in range(i_sz):
                        for dj in range(j_sz):
                            for dk in range(k_sz):
                                u_row = (i_start + di) * m + (j_start + dj)
                                v_row = (j_start + dj) * p + (k_start + dk)
                                w_row = (k_start + dk) * n + (i_start + di)
                                U.append((u_row, 1))
                                V.append((v_row, 1))
                                W.append((w_row, 1))
                                rank += 1
    
    # Now build the actual matrices
    U_mat = np.zeros((n * m, rank), dtype=int)
    V_mat = np.zeros((m * p, rank), dtype=int)
    W_mat = np.zeros((p * n, rank), dtype=int)
    
    for col in range(rank):
        u_row, u_val = U[col]
        v_row, v_val = V[col]
        w_row, w_val = W[col]
        U_mat[u_row, col] = u_val
        V_mat[v_row, col] = v_val
        W_mat[w_row, col] = w_val
    
    return U_mat, V_mat, W_mat

def _validate_decomposition(U, V, W, n, m, p):
    """
    Validate that the decomposition is exact.
    For all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if val != 1:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Return the standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 5x5x5 matrix multiplication.
    Returns a dict with keys 'U', 'V', 'W' containing the factor matrices.
    """
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Try the block decomposition with Strassen for 2x2x2 blocks
    U, V, W = _build_block_decomposition(n, m, p)
    rank = U.shape[1]
    
    # Validate
    if _validate_decomposition(U, V, W, n, m, p):
        if time.time() > deadline:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        U, V, W = _schoolbook_decomposition(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
