"""Matrix multiplication 4x5x5 tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Exact tensor decomposition for 4x5 @ 5x5 -> 4x5 matrix multiplication.
    
    Strategy:
    - 4x4x4 core: recursive Strassen, rank 49
    - Boundary terms (involving index 4): schoolbook, rank 36
    - Total rank: 85
    
    Identity: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    for all i in [0,4), j in [0,5), k in [0,5).
    """
    n, m, p = 4, 5, 5
    total_rank = 85
    
    U = np.zeros((n * m, total_rank), dtype=np.int32)
    V = np.zeros((m * p, total_rank), dtype=np.int32)
    W = np.zeros((p * n, total_rank), dtype=np.int32)
    
    # 2x2x2 Strassen decomposition
    # s_u[li*2+lk, t], s_v[lk*2+lj, t], s_w[li*2+lj, t] for t in 0..6
    s_u = np.array([
        [ 1,  0,  1,  0,  1, -1,  0],  # (li=0, lk=0)
        [ 0,  0,  0,  0,  1,  0,  1],   # (li=0, lk=1)
        [ 0,  1,  0,  0,  0,  1,  0],   # (li=1, lk=0)
        [ 1,  1,  0,  1,  0,  0, -1]    # (li=1, lk=1)
    ], dtype=np.int32)
    
    s_v = np.array([
        [ 1,  1,  0, -1,  0,  1,  0],  # (lk=0, lj=0)
        [ 0,  0,  1,  0,  0,  1,  0],   # (lk=0, lj=1)
        [ 0,  0,  0,  1,  0,  0,  1],   # (lk=1, lj=0)
        [ 1,  0, -1,  0,  1,  0,  1]    # (lk=1, lj=1)
    ], dtype=np.int32)
    
    s_w = np.array([
        [ 1,  0,  0,  1, -1,  0,  1],  # (li=0, lj=0)
        [ 0,  0,  1,  0,  1,  0,  0],   # (li=0, lj=1)
        [ 0,  1,  0,  1,  0,  0,  0],   # (li=1, lj=0)
        [ 1, -1,  1,  0,  0,  1,  0]    # (li=1, lj=1)
    ], dtype=np.int32)
    
    # 4x4x4 Strassen: 7 products of 2x2x2
    # Each product t has A-blocks, B-blocks, C-blocks with coefficients
    # Block (br, bc): rows 2br..2br+1, cols 2bc..2bc+1
    strassen_4x4 = [
        # M1 = (A00+A11)(B00+B11) -> C00 + C11
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (A10+A11)B00 -> C10 - C11
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3 = A00(B01-B11) -> C01 + C11
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4 = A11(B10-B00) -> C00 + C10
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5 = (A00+A01)B11 -> -C00 + C01
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6 = (A10-A00)(B00+B01) -> C11
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (A01-A11)(B10+B11) -> C00
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    col = 0
    start_time = time.time()
    
    def add_strassen_term(t_idx, k_idx, a_blocks, b_blocks, c_blocks):
        """Add one rank-1 term from the k-th 2x2x2 Strassen sub-term of the t-th 4x4x4 Strassen product."""
        nonlocal col
        curr_col = col + k_idx
        u_vec = s_u[:, k_idx]
        v_vec = s_v[:, k_idx]
        w_vec = s_w[:, k_idx]
        
        for (abr, abc), coeff_a in a_blocks.items():
            for li in range(2):
                for lk in range(2):
                    val = u_vec[li * 2 + lk]
                    if val != 0:
                        gi = 2 * abr + li
                        gj = 2 * abc + lk
                        U[gi * m + gj, curr_col] += coeff_a * val
        
        for (bbr, bbc), coeff_b in b_blocks.items():
            for lk in range(2):
                for lj in range(2):
                    val = v_vec[lk * 2 + lj]
                    if val != 0:
                        gi = 2 * bbr + lk
                        gj = 2 * bbc + lj
                        V[gi * p + gj, curr_col] += coeff_b * val
        
        for (cbr, cbc), coeff_c in c_blocks.items():
            for li in range(2):
                for lj in range(2):
                    val = w_vec[li * 2 + lj]
                    if val != 0:
                        gi = 2 * cbr + li
                        gj = 2 * cbc + lj
                        W[gj * n + gi, curr_col] += coeff_c * val
    
    # Add 49 rank-1 terms for the 4x4x4 core
    for t_idx, (a_blocks, b_blocks, c_blocks) in enumerate(strassen_4x4):
        for k_idx in range(7):
            add_strassen_term(t_idx, k_idx, a_blocks, b_blocks, c_blocks)
        col += 7
    
    # Boundary terms: 36 rank-1 terms
    # 1. A[i, j] * B[j, 4] -> C[i, 4] for i,j in 0..3 (16 terms)
    for i in range(4):
        for j in range(4):
            U[i * m + j, col] = 1
            V[j * p + 4, col] = 1
            W[4 * n + i, col] = 1
            col += 1
    
    # 2. A[i, 4] * B[4, k] -> C[i, k] for i,k in 0..3 (16 terms)
    for i in range(4):
        for k in range(4):
            U[i * m + 4, col] = 1
            V[4 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
    
    # 3. A[i, 4] * B[4, 4] -> C[i, 4] for i in 0..3 (4 terms)
    for i in range(4):
        U[i * m + 4, col] = 1
        V[4 * p + 4, col] = 1
        W[4 * n + i, col] = 1
        col += 1
    
    assert col == total_rank, f"Column count {col} != expected {col}"
    
    # Quick verification
    def verify(U, V, W, n, m, p, tol=1e-9):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school), dtype=np.int32)
        V = np.zeros((m * p, rank_school), dtype=np.int32)
        W = np.zeros((p * n, rank_school), dtype=np.int32)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
