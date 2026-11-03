"""Tensor decomposition for 4x4 x 4x9 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Returns U, V, W for M_{2,2,2} using Strassen's algorithm.
    Rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))

    # Strassen formulas:
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)

    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # C indices: 0=C00, 1=C01, 2=C10, 3=C11

    # M1: A00+A11, B00+B11
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # M2: A10+A11, B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1

    # M3: A00, B01-B11
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1

    # M4: A11, B10-B00
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[2, 3] = 1

    # M5: A00+A01, B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1

    # M6: A10-A00, B00+B01
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # M7: A01-A11, B10+B11
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W

def _decomp_4x4x4():
    """
    Decomposition of M_{4,4,4} using 2-level Strassen.
    Rank 49.
    U: (16, 49), V: (16, 49), W: (16, 49)
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))

    inner_U, inner_V, inner_W = _strassen_2x2()

    # Outer Strassen coefficients
    outer_A = [
        [1, 0, 0, 1],   # M1: A00+A11
        [0, 0, 1, 1],   # M2: A10+A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00+A01
        [0, 0, 1, -1],  # M6: A10-A00
        [0, 1, 0, -1]   # M7: A01-A11
    ]

    outer_B = [
        [1, 0, 0, 1],   # M1: B00+B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01-B11
        [0, 0, 1, -1],  # M4: B10-B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00+B01
        [0, 0, 1, 1]    # M7: B10+B11
    ]

    outer_C = [
        [1, 0, 0, 1],   # M1 -> C00, C11
        [0, 0, 1, -1],  # M2 -> C10, C11(-1)
        [0, 1, 0, 1],   # M3 -> C01, C11
        [-1, 0, 1, 0],  # M4 -> C00(-1), C10
        [1, 1, 0, 0],   # M5 -> C00, C01
        [0, 0, 0, 1],   # M6 -> C11
        [1, 0, 0, 0]    # M7 -> C00
    ]

    for t in range(7):
        col_offset = t * 7
        for r in range(7):
            col = col_offset + r
            for li in range(2):
                for lj in range(2):
                    b = li * 2 + lj
                    coeff_U = outer_A[t][b] * inner_U[li*2 + lj, r]
                    coeff_V = outer_B[t][b] * inner_V[li*2 + lj, r]
                    coeff_W = outer_C[t][b] * inner_W[li*2 + lj, r]

                    for ei in range(2):
                        for ej in range(2):
                            g_i = 2 * (b // 2) + ei
                            g_j = 2 * (b % 2) + ej
                            U[g_i * 4 + g_j, col] = coeff_U
                            V[g_j * 4 + g_i, col] = coeff_V
                            W[g_i * 4 + g_j, col] = coeff_W

    return U, V, W

def _decomp_4x4x2():
    """
    Decomposition of M_{4,4,2} using 2-level Strassen restricted to 2 output columns.
    We multiply a 4x4 A by a 4x2 B to get a 4x2 C.
    Rank: 2 * 49 = 98 is too high. Instead, treat the two output columns as
    two independent 4x4x1 multiplications, each of rank 16 (schoolbook) -> 32.
    But we can do better: use Strassen on the A side (4x4) but B has only 2 cols.
    
    Approach: split A into 2x2 blocks of 2x2. Split B (4x2) into 2x2 blocks of
    2x1. The result C (4x2) splits into 2x2 blocks of 2x1.
    For each pair of A-block (2x2) and B-block (2x1), we need a 2x2 x 2x1 -> 2x1
    multiplication. That's M_{2,2,1} which has rank 2 (two dot products of 2-vectors).
    There are 2 (row-blocks of A) * 2 (col-blocks of A) = 4 such subproblems, each
    rank 2, total rank 8. But we can use Strassen-like on the 2x2 A blocks.
    
    Actually, the cleanest: M_{4,4,2}. Split A into 4 blocks of 2x2, B into 2 blocks
    of 2x2 (each 2x2 but only 2 cols total means B is 4x2 = two 2x2 blocks stacked...
    no, B is 4x2, split rows into two 2x2). C is 4x2, split into two 2x2.
    C_top = A00*B00 + A01*B10, C_bot = A10*B00 + A11*B10.
    Each term is 2x2 * 2x2 -> 2x2, rank 7 (Strassen). 4 terms * 7 = 28.
    Total rank 28.
    """
    n, m, p = 4, 4, 2
    # 4 subproblems, each 2x2x2 rank 7 -> total 28
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))

    inner_U, inner_V, inner_W = _strassen_2x2()

    # A blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3),
    #           A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    # B blocks: B00 (rows 0-1, cols 0-1), B10 (rows 2-3, cols 0-1)
    # C blocks: C00 (rows 0-1, cols 0-1), C10 (rows 2-3, cols 0-1)
    # C00 = A00*B00 + A01*B10
    # C10 = A10*B00 + A11*B10

    # Each term: (A_block 2x2) * (B_block 2x2) -> (C_block 2x2)
    # We have 4 terms, each uses the 7-term Strassen inner decomp.
    terms = [
        # (A_block_row, A_block_col, B_block_row, C_block_row, sign)
        (0, 0, 0, 0, 1),   # A00*B00 -> C00
        (0, 1, 1, 0, 1),   # A01*B10 -> C00
        (1, 0, 0, 1, 1),   # A10*B00 -> C10
        (1, 1, 1, 1, 1),   # A11*B10 -> C10
    ]

    for t_idx, (ar, ac, br, cr, sign) in enumerate(terms):
        col_offset = t_idx * 7
        for r in range(7):
            col = col_offset + r
            # A_block at (ar, ac): rows 2*ar..2*ar+1, cols 2*ac..2*ac+1
            for ei in range(2):
                for ej in range(2):
                    g_i = 2 * ar + ei
                    g_j = 2 * ac + ej
                    U[g_i * 4 + g_j, col] = sign * inner_U[ei * 2 + ej, r]
            # B_block at row br: rows 2*br..2*br+1, cols 0-1 (since p=2, cols are 0,1)
            for ei in range(2):
                for ej in range(2):
                    g_j = 2 * br + ei   # row within B
                    g_k = ej            # col within B (0 or 1)
                    V[g_j * 2 + g_k, col] = sign * inner_V[ei * 2 + ej, r]
            # C_block at row cr: rows 2*cr..2*cr+1, cols 0-1
            for ei in range(2):
                for ej in range(2):
                    g_k = ej            # col within C (0 or 1)
                    g_i = 2 * cr + ei   # row within C
                    W[g_k * 4 + g_i, col] = sign * inner_W[ei * 2 + ej, r]

    return U, V, W

def _schoolbook_nm_p(n, m, p):
    """
    Schoolbook decomposition for M_{n,m,p}.
    Rank n*m*p.
    """
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _decomp_4x4x9_combined():
    """
    Decomposition of M_{4,4,9} by splitting into M_{4,4,4} + M_{4,4,4} + M_{4,4,1}.
    Total rank: 49 + 49 + 16 = 114.
    """
    n, m, p = 4, 4, 9
    total_rank = 49 + 49 + 16

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    U444, V444, W444 = _decomp_4x4x4()
    U441, V441, W441 = _schoolbook_nm_p(4, 4, 1)

    # First 49 cols: 4x4x4 for k=0..3
    U[:, :49] = U444
    for j in range(m):
        for k in range(4):
            V[j * 9 + k, :49] = V444[j * 4 + k, :]
    for k in range(4):
        for i in range(n):
            W[k * 4 + i, :49] = W444[k * 4 + i, :]

    # Next 49 cols: 4x4x4 for k=4..7
    U[:, 49:98] = U444
    for j in range(m):
        for k in range(4):
            V[j * 9 + (4 + k), 49:98] = V444[j * 4 + k, :]
    for k in range(4):
        for i in range(n):
            W[(4 + k) * 4 + i, 49:98] = W444[k * 4 + i, :]

    # Last 16 cols: 4x4x1 for k=8
    U[:, 98:114] = U441
    for j in range(m):
        V[j * 9 + 8, 98:114] = V441[j * 1 + 0, :]
    for i in range(n):
        W[8 * 4 + i, 98:114] = W441[0 * 4 + i, :]

    return U, V, W

def _decomp_4x4x9_v2():
    """
    Decomposition of M_{4,4,9} by splitting into M_{4,4,4} + M_{4,4,2} + M_{4,4,2} + M_{4,4,1}.
    Total rank: 49 + 28 + 28 + 16 = 121. (Worse than 114, kept for comparison.)
    """
    n, m, p = 4, 4, 9
    total_rank = 49 + 28 + 28 + 16

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    U444, V444, W444 = _decomp_4x4x4()
    U442, V442, W442 = _decomp_4x4x2()
    U441, V441, W441 = _schoolbook_nm_p(4, 4, 1)

    # First 49 cols: 4x4x4 for k=0..3
    U[:, :49] = U444
    for j in range(m):
        for k in range(4):
            V[j * 9 + k, :49] = V444[j * 4 + k, :]
    for k in range(4):
        for i in range(n):
            W[k * 4 + i, :49] = W444[k * 4 + i, :]

    # Next 28 cols: 4x4x2 for k=4..5
    U[:, 49:77] = U442
    for j in range(m):
        for k in range(2):
            V[j * 9 + (4 + k), 49:77] = V442[j * 2 + k, :]
    for k in range(2):
        for i in range(n):
            W[(4 + k) * 4 + i, 49:77] = W442[k * 4 + i, :]

    # Next 28 cols: 4x4x2 for k=6..7
    U[:, 77:105] = U442
    for j in range(m):
        for k in range(2):
            V[j * 9 + (6 + k), 77:105] = V442[j * 2 + k, :]
    for k in range(2):
        for i in range(n):
            W[(6 + k) * 4 + i, 77:105] = W442[k * 4 + i, :]

    # Last 16 cols: 4x4x1 for k=8
    U[:, 105:121] = U441
    for j in range(m):
        V[j * 9 + 8, 105:121] = V441[j * 1 + 0, :]
    for i in range(n):
        W[8 * 4 + i, 105:121] = W441[0 * 4 + i, :]

    return U, V, W

def _verify_decomp(U, V, W, n, m, p, atol=1e-9):
    """Vectorized verification of the decomposition."""
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)

    # Compute sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] for all (i,j,k)
    # Use einsum for efficiency
    # U_r: (n,m,r), V_r: (m,p,r), W_r: (p,n,r)
    # We want: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # = einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    result = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r, optimize=True)
    return np.allclose(result, 1.0, atol=atol)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin: 150s limit, keep 20s buffer

    # Primary approach: 4x4x4 + 4x4x4 + 4x4x1 = rank 49 + 49 + 16 = 114
    U, V, W = _decomp_4x4x9_combined()

    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        if _verify_decomp(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook (always valid, rank n*m*p = 144)
    Us, Vs, Ws = _schoolbook_nm_p(n, m, p)
    return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}

# EVOLVE-BLOCK-END
