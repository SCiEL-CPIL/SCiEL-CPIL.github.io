import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The tensor T[u, v, w] = sum_r U[u,r]*V[v,r]*W[w,r] should equal 1
    iff there exist i,k,j with u=i*m+k, v=k*p+j, w=j*n+i (and consistent i,k,j).
    """
    rank = U.shape[1]
    # Build target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                u = i * m + k
                v = k * p + j
                w = j * n + i
                T_target[u, v, w] = 1.0
    
    # Compute decomposition tensor
    T_decomp = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T_decomp += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
    
    error = np.max(np.abs(T_decomp - T_target))
    return error < tol, error

def _strassen_block(U, V, W, col, a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, m, p, n):
    """
    Adds 7 rank-1 terms corresponding to a Strassen decomposition of a 2x2x2 block.
    """
    u_a = a_rows[0] * m + a_cols[0]
    u_b = a_rows[0] * m + a_cols[1]
    u_c = a_rows[1] * m + a_cols[0]
    u_d = a_rows[1] * m + a_cols[1]
    
    v_e = b_rows[0] * p + b_cols[0]
    v_f = b_rows[0] * p + b_cols[1]
    v_g = b_rows[1] * p + b_cols[0]
    v_h = b_rows[1] * p + b_cols[1]
    
    w_00 = c_cols[0] * n + c_rows[0]
    w_01 = c_cols[1] * n + c_rows[0]
    w_10 = c_cols[0] * n + c_rows[1]
    w_11 = c_cols[1] * n + c_rows[1]
    
    # M1: (a+d)(e+h)
    U[u_a, col] += 1.0
    U[u_d, col] += 1.0
    V[v_e, col] += 1.0
    V[v_h, col] += 1.0
    W[w_00, col] += 1.0
    W[w_11, col] += 1.0
    col += 1
    
    # M2: (c+d)e
    U[u_c, col] += 1.0
    U[u_d, col] += 1.0
    V[v_e, col] += 1.0
    W[w_10, col] += 1.0
    W[w_11, col] += -1.0
    col += 1
    
    # M3: a(f-h)
    U[u_a, col] += 1.0
    V[v_f, col] += 1.0
    V[v_h, col] += -1.0
    W[w_01, col] += 1.0
    W[w_11, col] += 1.0
    col += 1
    
    # M4: d(g-e)
    U[u_d, col] += 1.0
    V[v_g, col] += 1.0
    V[v_e, col] += -1.0
    W[w_00, col] += 1.0
    W[w_10, col] += 1.0
    col += 1
    
    # M5: (a+b)h
    U[u_a, col] += 1.0
    U[u_b, col] += 1.0
    V[v_h, col] += 1.0
    W[w_00, col] += -1.0
    W[w_01, col] += 1.0
    col += 1
    
    # M6: (c-a)(e+f)
    U[u_c, col] += 1.0
    U[u_a, col] += -1.0
    V[v_e, col] += 1.0
    V[v_f, col] += 1.0
    W[w_11, col] += 1.0
    col += 1
    
    # M7: (b-d)(g+h)
    U[u_b, col] += 1.0
    U[u_d, col] += -1.0
    V[v_g, col] += 1.0
    V[v_h, col] += 1.0
    W[w_00, col] += 1.0
    col += 1
    
    return col

def _construct_rank_44():
    """
    Construct the rank-44 decomposition using Strassen blocks for 3x4x8.
    Split B into two 4x4 halves. For each half, use:
    - Strassen (7 terms) for A[0:2,0:2] x B_half[0:2,0:2] -> C[0:2,0:2]
    - Strassen (7 terms) for A[0:2,0:2] x B_half[0:2,2:4] -> C[0:2,2:4]
    - Strassen (7 terms) for A[0:2,2:4] x B_half[2:4,0:2] -> C[0:2,0:2]
    - Strassen (7 terms) for A[0:2,2:4] x B_half[2:4,2:4] -> C[0:2,2:4]
    - Rank-4 for A[2,:] x B_half[0:2,:] -> C[2,0:2]
    - Rank-4 for A[2,:] x B_half[2:4,:] -> C[2,2:4]
    Total per 3x4x4: 7+7+7+7+4+4 = 36? No wait...
    
    Actually, let me reconsider. The Strassen block computes a 2x2x2 tensor product.
    For A[0:2, k1:k1+2] x B[k1:k1+2, j1:j1+2] -> C[0:2, j1:j1+2], that's 7 terms.
    
    For 3x4x4:
    - A[0:2, 0:2] x B[0:2, 0:2] -> C[0:2, 0:2]: 7 terms
    - A[0:2, 0:2] x B[0:2, 2:4] -> C[0:2, 2:4]: 7 terms
    - A[0:2, 2:4] x B[2:4, 0:2] -> C[0:2, 0:2]: 7 terms
    - A[0:2, 2:4] x B[2:4, 2:4] -> C[0:2, 2:4]: 7 terms
    - A[2, 0:2] x B[0:2, 0:2] -> C[2, 0:2]: 4 terms
    - A[2, 0:2] x B[0:2, 2:4] -> C[2, 2:4]: 4 terms
    - A[2, 2:4] x B[2:4, 0:2] -> C[2, 0:2]: 4 terms
    - A[2, 2:4] x B[2:4, 2:4] -> C[2, 2:4]: 4 terms
    Total: 28 + 16 = 44 for 3x4x4.
    
    For 3x4x8, we have two such blocks (j=0:4 and j=4:8):
    44 * 2 = 88. That's the same as before!
    
    I made an error. Let me recount.
    """
    n, m, p = DIMENSIONS  # 3, 4, 8
    rank = 88
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # For each half of B (columns 0:4 and 4:8):
    for j_offset in [0, 4]:
        # A[0:2, 0:2] x B[0:2, j_offset:j_offset+2] -> C[0:2, j_offset:j_offset+2]
        col = _strassen_block(
            U, V, W, col,
            a_rows=[0, 1], a_cols=[0, 1],
            b_rows=[0, 1], b_cols=[j_offset, j_offset+1],
            c_rows=[0, 1], c_cols=[j_offset, j_offset+1],
            m=m, p=p, n=n
        )
        # A[0:2, 0:2] x B[0:2, j_offset+2:j_offset+4] -> C[0:2, j_offset+2:j_offset+4]
        col = _strassen_block(
            U, V, W, col,
            a_rows=[0, 1], a_cols=[0, 1],
            b_rows=[0, 1], b_cols=[j_offset+2, j_offset+3],
            c_rows=[0, 1], c_cols=[j_offset+2, j_offset+3],
            m=m, p=p, n=n
        )
        # A[0:2, 2:4] x B[2:4, j_offset:j_offset+2] -> C[0:2, j_offset:j_offset+2]
        col = _strassen_block(
            U, V, W, col,
            a_rows=[0, 1], a_cols=[2, 3],
            b_rows=[2, 3], b_cols=[j_offset, j_offset+1],
            c_rows=[0, 1], c_cols=[j_offset, j_offset+1],
            m=m, p=p, n=n
        )
        # A[0:2, 2:4] x B[2:4, j_offset+2:j_offset+4] -> C[0:2, j_offset+2:j_offset+4]
        col = _strassen_block(
            U, V, W, col,
            a_rows=[0, 1], a_cols=[2, 3],
            b_rows=[2, 3], b_cols=[j_offset+2, j_offset+3],
            c_rows=[0, 1], c_cols=[j_offset+2, j_offset+3],
            m=m, p=p, n=n
        )
        # A[2, 0:2] x B[0:2, j_offset:j_offset+2] -> C[2, j_offset:j_offset+2]
        for k in [0, 1]:
            for j in [j_offset, j_offset+1]:
                U[2 * m + k, col] = 1.0
                V[k * p + j, col] = 1.0
                W[j * n + 2, col] = 1.0
                col += 1
        # A[2, 0:2] x B[0:2, j_offset+2:j_offset+4] -> C[2, j_offset+2:j_offset+4]
        for k in [0, 1]:
            for j in [j_offset+2, j_offset+3]:
                U[2 * m + k, col] = 1.0
                V[k * p + j, col] = 1.0
                W[j * n + 2, col] = 1.0
                col += 1
        # A[2, 2:4] x B[2:4, j_offset:j_offset+2] -> C[2, j_offset:j_offset+2]
        for k in [2, 3]:
            for j in [j_offset, j_offset+1]:
                U[2 * m + k, col] = 1.0
                V[k * p + j, col] = 1.0
                W[j * n + 2, col] = 1.0
                col += 1
        # A[2, 2:4] x B[2:4, j_offset+2:j_offset+4] -> C[2, j_offset+2:j_offset+4]
        for k in [2, 3]:
            for j in [j_offset+2, j_offset+3]:
                U[2 * m + k, col] = 1.0
                V[k * p + j, col] = 1.0
                W[j * n + 2, col] = 1.0
                col += 1
    
    if col != rank:
        raise Exception(f"Rank mismatch: expected {rank}, got {col}")
    
    valid, error = _verify_decomposition(U, V, W, n, m, p)
    if not valid:
        raise Exception(f"Verification failed with error {error}")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _construct_rank_88():
    """
    Construct the rank-88 decomposition using Strassen blocks.
    C = A * B where A is 3x4, B is 4x8, C is 3x8.
    """
    n, m, p = DIMENSIONS  # 3, 4, 8
    rank = 88
    U = np.zeros((n * m, rank), dtype=float)  # (12, 88)
    V = np.zeros((m * p, rank), dtype=float)  # (32, 88)
    W = np.zeros((p * n, rank), dtype=float)  # (24, 88)
    
    col = 0
    
    # Part 1a: A[0:2, 0:2] * B[0:2, :]
    # B[0:2, :] has columns 0..7, split into pairs: (0,1), (2,3), (4,5), (6,7)
    for k_start in range(0, 8, 2):
        col = _strassen_block(
            U, V, W, col,
            a_rows=[0, 1], a_cols=[0, 1],
            b_rows=[0, 1], b_cols=[k_start, k_start+1],
            c_rows=[0, 1], c_cols=[k_start, k_start+1],
            m=m, p=p, n=n
        )
    
    # Part 1b: A[0:2, 2:4] * B[2:4, :]
    for k_start in range(0, 8, 2):
        col = _strassen_block(
            U, V, W, col,
            a_rows=[0, 1], a_cols=[2, 3],
            b_rows=[2, 3], b_cols=[k_start, k_start+1],
            c_rows=[0, 1], c_cols=[k_start, k_start+1],
            m=m, p=p, n=n
        )
    
    # Part 2: A[2:3, :] * B[:, :]
    # A[2, k] * B[k, j] for k=0..3, j=0..7
    for k in range(4):
        for j in range(8):
            U[2 * m + k, col] = 1.0  # A[2, k]
            V[k * p + j, col] = 1.0  # B[k, j]
            W[j * n + 2, col] = 1.0  # C[2, j]
            col += 1
    
    if col != rank:
        raise Exception(f"Rank mismatch: expected {rank}, got {col}")
    
    # Verify
    valid, error = _verify_decomposition(U, V, W, n, m, p)
    if not valid:
        raise Exception(f"Verification failed with error {error}")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _strassen_row_block(U, V, W, col, a_row, b_rows, c_cols, m, p, n):
    """
    For a single row of A (row a_row) and 2 rows of B (b_rows[0], b_rows[1]),
    compute the contribution to 2 columns of C (c_cols[0], c_cols[1]) using 3 rank terms
    instead of 4.
    
    We want: A[a_row, b0]*B[b0, c0] + A[a_row, b0]*B[b0, c1] + A[a_row, b1]*B[b1, c0] + A[a_row, b1]*B[b1, c1]
    But wait, this isn't quite right. The Strassen trick applies when we have a 2x2 matrix product.
    
    Actually, for a single row of A and 2 rows of B, the contribution to C is:
    C[a_row, c0] += A[a_row, b0]*B[b0, c0] + A[a_row, b1]*B[b1, c0]
    C[a_row, c1] += A[a_row, b0]*B[b0, c1] + A[a_row, b1]*B[b1, c1]
    
    This is a 1x2 times 2x2 product. We can use:
    M1 = (A[a_row,b0] + A[a_row,b1]) * (B[b0,c0] + B[b1,c0])
    M2 = A[a_row,b0] * B[b1,c0]
    M3 = A[a_row,b1] * B[b0,c0]
    Then C[a_row,c0] = M1 - M2 - M3
    
    Similarly for c1. But this gives 3 terms per c-column, so 6 terms for 2 c-columns,
    which is the same as the naive 4 terms (2 k * 2 j). No savings!
    
    The Strassen trick only saves when we have a 2x2 x 2x2 product, not 1x2 x 2x2.
    """
    pass

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Use the rank-88 solution
    result = _construct_rank_88()
    
    return result
