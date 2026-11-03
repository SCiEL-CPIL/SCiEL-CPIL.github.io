"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time
import random
import itertools

def _schoolbook(n, m, p):
    """Return (U, V, W) for schoolbook multiplication of n x m by m x p.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank) with W indexed by k*n + i."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W


def _strassen_222():
    """Return (U, V, W) for 2x2x2 Strassen.
    A: 2x2, B: 2x2, C: 2x2.
    U indexed by (i,j) -> i*2+j, V by (j,k) -> j*2+k, W by (k,i) -> k*2+i.
    Strassen intermediates:
      M1 = (A00+A11)(B00+B11)
      M2 = (A10+A11)B00
      M3 = A00(B01-B11)
      M4 = A11(B10-B00)
      M5 = (A00+A01)B11
      M6 = (A10-A00)(B00+B01)
      M7 = (A01-A11)(B10+B11)
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1  # C00 (k=0,i=0)
    W[3][0] = 1  # C11 (k=1,i=1)
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1  # C10 (k=1,i=0)
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1  # C01 (k=0,i=1)
    
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1  # C00 (k=0,i=0)
    W[2][3] = 1  # C10 (k=1,i=0)
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1 # C00
    W[1][4] = 1  # C01
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # C11 (k=1,i=1)
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # C00 (k=0,i=0)
    
    return U, V, W


def _schoolbook_212():
    """Return (U, V, W) for 2x1x2 schoolbook.
    A: 2x1, B: 1x2, C: 2x2.
    U indexed by (i,j) -> i*1+j = i, V by (j,k) -> j*2+k = k, W by (k,i) -> k*2+i.
    C00 = A0*B0, C01 = A0*B1, C10 = A1*B0, C11 = A1*B1.
    """
    rank = 4
    U = [[0] * rank for _ in range(2)]
    V = [[0] * rank for _ in range(2)]
    W = [[0] * rank for _ in range(4)]
    
    # A0*B0 -> C00
    U[0][0] = 1
    V[0][0] = 1
    W[0][0] = 1  # k=0, i=0
    
    # A0*B1 -> C01
    U[0][1] = 1
    V[1][1] = 1
    W[1][1] = 1  # k=0, i=1
    
    # A1*B0 -> C10
    U[1][2] = 1
    V[0][2] = 1
    W[2][2] = 1  # k=1, i=0
    
    # A1*B1 -> C11
    U[1][3] = 1
    V[1][3] = 1
    W[3][3] = 1  # k=1, i=1
    
    return U, V, W


def _schoolbook_221():
    """Return (U, V, W) for 2x2x1 schoolbook.
    A: 2x2, B: 2x1, C: 2x1.
    U indexed by (i,j) -> i*2+j, V by (j,k) -> j*1+k = j, W by (k,i) -> k*2+i = i.
    C0 = A00*B0 + A01*B1, C1 = A10*B0 + A11*B1.
    """
    rank = 4
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(2)]
    W = [[0] * rank for _ in range(2)]
    
    # A00*B0 -> C0
    U[0][0] = 1
    V[0][0] = 1
    W[0][0] = 1
    
    # A01*B1 -> C0
    U[1][1] = 1
    V[1][1] = 1
    W[0][1] = 1
    
    # A10*B0 -> C1
    U[2][2] = 1
    V[0][2] = 1
    W[1][2] = 1
    
    # A11*B1 -> C1
    U[3][3] = 1
    V[1][3] = 1
    W[1][3] = 1
    
    return U, V, W


def _schoolbook_112():
    """Return (U, V, W) for 1x1x2 schoolbook.
    A: 1x1, B: 1x2, C: 1x2.
    U indexed by (i,j) -> 0, V by (j,k) -> k, W by (k,i) -> k*1+0 = k.
    C0 = A*B0, C1 = A*B1.
    """
    rank = 2
    U = [[0] * rank for _ in range(1)]
    V = [[0] * rank for _ in range(2)]
    W = [[0] * rank for _ in range(2)]
    
    U[0][0] = 1
    V[0][0] = 1
    W[0][0] = 1
    
    U[0][1] = 1
    V[1][1] = 1
    W[1][1] = 1
    
    return U, V, W


def _concat_decompositions(n, m, p, decomp_list, u_offsets, v_offsets, w_offsets):
    """Concatenate multiple decompositions into a single global one.
    decomp_list: list of (U_i, V_i, W_i)
    u_offsets: list of (row_start, col_start) for each U_i in global U
    v_offsets: list of (row_start, col_start) for each V_i in global V
    w_offsets: list of (row_start, col_start) for each W_i in global W
    """
    total_rank = 0
    for U_i, V_i, W_i in decomp_list:
        total_rank += len(U_i[0]) if U_i else 0
    
    global_U = [[0] * total_rank for _ in range(n * m)]
    global_V = [[0] * total_rank for _ in range(m * p)]
    global_W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for idx, (U_i, V_i, W_i) in enumerate(decomp_list):
        rank_i = len(U_i[0]) if U_i else 0
        u_row_start, u_col_start = u_offsets[idx]
        v_row_start, v_col_start = v_offsets[idx]
        w_row_start, w_col_start = w_offsets[idx]
        
        for r in range(len(U_i)):
            for c in range(rank_i):
                val = U_i[r][c]
                if val != 0:
                    global_row = u_row_start + r
                    global_col = col_offset + c
                    if 0 <= global_row < n*m and 0 <= global_col < total_rank:
                        global_U[global_row][global_col] += val
        for r in range(len(V_i)):
            for c in range(rank_i):
                val = V_i[r][c]
                if val != 0:
                    global_row = v_row_start + r
                    global_col = col_offset + c
                    if 0 <= global_row < m*p and 0 <= global_col < total_rank:
                        global_V[global_row][global_col] += val
        for r in range(len(W_i)):
            for c in range(rank_i):
                val = W_i[r][c]
                if val != 0:
                    global_row = w_row_start + r
                    global_col = col_offset + c
                    if 0 <= global_row < p*n and 0 <= global_col < total_rank:
                        global_W[global_row][global_col] += val
        col_offset += rank_i
    
    return global_U, global_V, global_W


def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 2, 3, 7
    
    # Decompose 2x3x7 into:
    # A (2x3) * B (3x7)
    # Split B into B1 (3x3) and B2 (3x4)
    # So we have:
    # 1. A * B1: 2x3x3
    # 2. A * B2: 2x3x4
    
    # For 2x3x3:
    # Split A into A1 (2x2) and A2 (2x1)
    # B1 is 3x3, split into B11 (3x1)? No, let's think differently.
    # Actually, let's just use schoolbook for simplicity first, then try to optimize.
    
    # Let's try:
    # 2x3x3: 
    #   A: 2x3, B: 3x3
    #   Split A into A1 (2x2) and A2 (2x1)
    #   B stays 3x3
    #   A1*B: 2x2x3 -> split B into B1 (2x3)? No, B is 3x3.
    #   Let's split B into B1 (3x2) and B2 (3x1)? No, B is 3x3, so columns.
    #   B: 3 rows, 3 cols. Split cols into 2 and 1.
    #   B1: 3x2, B2: 3x1
    #   A1*B1: 2x2x2 -> Strassen (rank 7)
    #   A1*B2: 2x2x1 -> Schoolbook (rank 4)
    #   A2*B1: 2x1x2 -> Schoolbook (rank 4)
    #   A2*B2: 2x1x1 -> Schoolbook (rank 2)
    #   Total for 2x3x3: 7 + 4 + 4 + 2 = 17
    
    # For 2x3x4:
    #   A: 2x3, B: 3x4
    #   Split A into A1 (2x2) and A2 (2x1)
    #   Split B into B1 (3x2) and B2 (3x2)
    #   A1*B1: 2x2x2 -> Strassen (rank 7)
    #   A1*B2: 2x2x2 -> Strassen (rank 7)
    #   A2*B1: 2x1x2 -> Schoolbook (rank 4)
    #   A2*B2: 2x1x2 -> Schoolbook (rank 4)
    #   Total for 2x3x4: 7 + 7 + 4 + 4 = 22
    
    # Total: 17 + 22 = 39
    
    # Let's verify this is correct and construct the matrices.
    
    # First, let's define the sub-problems and their decompositions.
    
    # 2x3x3 decomposition:
    # A: 2x3, rows 0,1; cols 0,1,2
    # A1: rows 0,1; cols 0,1 (2x2)
    # A2: rows 0,1; col 2 (2x1)
    
    # B1: 3x3, rows 0,1,2; cols 0,1,2
    # B1a: rows 0,1,2; cols 0,1 (3x2)
    # B1b: rows 0,1,2; col 2 (3x1)
    
    # Sub-problems for 2x3x3:
    # 1. A1 * B1a: 2x2x2 (Strassen)
    #    A1: A[:, 0:2], B1a: B[0:3, 0:2]
    #    U indices: (i,j) in A1 -> i*2+j
    #    V indices: (j,k) in B1a -> j*2+k (but B1a has 2 cols, so k=0,1)
    #    Wait, B1a is 3x2, so V should be 3x2? No, V is (m*p) where m=2 (inner dim of A1*B1a), p=2 (cols of B1a).
    #    So V is 2*2=4? No, V is (m_sub * p_sub) = 2*2 = 4? 
    #    Actually, for sub-problem A_sub (n_sub x m_sub) * B_sub (m_sub x p_sub):
    #    U: n_sub * m_sub, V: m_sub * p_sub, W: p_sub * n_sub
    #    
    #    For A1 (2x2) * B1a (3x2): This doesn't match! A1 is 2x2, B1a is 3x2.
    #    The inner dimensions don't match: A1 has 2 cols, B1a has 3 rows.
    #    
    #    I made a mistake. Let's reconsider.
    #    
    #    A: 2x3 (rows: 2, cols: 3)
    #    B: 3x7 (rows: 3, cols: 7)
    #    
    #    When we split A into A1 (2x2) and A2 (2x1), we're splitting the columns of A.
    #    A1: A[:, 0:2] (2x2), A2: A[:, 2:3] (2x1)
    #    
    #    When we split B into B1 (3x3) and B2 (3x4), we're splitting the columns of B.
    #    B1: B[:, 0:3] (3x3), B2: B[:, 3:7] (3x4)
    #    
    #    Now, A1 * B1: (2x2) * (3x3) -> inner dims don't match! 2 != 3.
    #    
    #    The issue is that when we split A's columns and B's columns, the inner dimension (A's cols / B's rows) must match.
    #    
    #    Let's rethink the block structure.
    #    
    #    A: 2x3, B: 3x7
    #    
    #    We can split A's rows and B's cols, or A's cols and B's rows, or both.
    #    
    #    Option 1: Split A's cols (m=3 into 2+1) and B's rows (m=3 into 2+1).
    #    A1: A[:, 0:2] (2x2), A2: A[:, 2:3] (2x1)
    #    B1: B[0:2, :] (2x7), B2: B[2:3, :] (1x7)
    #    
    #    Then:
    #    A1 * B1: (2x2) * (2x7) -> 2x2x7
    #    A1 * B2: (2x2) * (1x7) -> inner dims 2 != 1, doesn't work.
    #    
    #    Option 2: Split A's rows (n=2 into 1+1) and B's cols (p=7 into 3+4).
    #    A1: A[0:1, :] (1x3), A2: A[1:2, :] (1x3)
    #    B1: B[:, 0:3] (3x3), B2: B[:, 3:7] (3x4)
    #    
    #    Then:
    #    A1 * B1: (1x3) * (3x3) -> 1x3x3
    #    A1 * B2: (1x3) * (3x4) -> 1x3x4
    #    A2 * B1: (1x3) * (3x3) -> 1x3x3
    #    A2 * B2: (1x3) * (3x4) -> 1x3x4
    #    
    #    This works! Each sub-problem has matching inner dims.
    #    
    #    Let's estimate ranks:
    #    1x3x3: schoolbook rank = 1*3*3 = 9
    #    1x3x4: schoolbook rank = 1*3*4 = 12
    #    Total: 9 + 12 + 9 + 12 = 42 (same as schoolbook, no improvement)
    #    
    #    We need to find a better block structure that allows using Strassen or other optimizations.
    #    
    #    Let's try:
    #    Split A's cols (m=3 into 2+1) and keep B whole.
    #    A1: A[:, 0:2] (2x2), A2: A[:, 2:3] (2x1)
    #    B: 3x7 (whole)
    #    
    #    A1 * B: (2x2) * (3x7) -> inner dims 2 != 3, doesn't work.
    #    
    #    Split B's rows (m=3 into 2+1) and keep A whole.
    #    B1: B[0:2, :] (2x7), B2: B[2:3, :] (1x7)
    #    A: 2x3 (whole)
    #    
    #    A * B1: (2x3) * (2x7) -> inner dims 3 != 2, doesn't work.
    #    
    #    It seems like we need to split both A's cols and B's rows in a way that the inner dims match.
    #    
    #    Let's try:
    #    Split A's cols into 1+1+1 and B's rows into 1+1+1.
    #    A1: A[:, 0:1] (2x1), A2: A[:, 1:2] (2x1), A3: A[:, 2:3] (2x1)
    #    B1: B[0:1, :] (1x7), B2: B[1:2, :] (1x7), B3: B[2:3, :] (1x7)
    #    
    #    Then:
    #    A1 * B1: (2x1) * (1x7) -> 2x1x7, rank = 2*1*7 = 14
    #    A1 * B2: (2x1) * (1x7) -> inner dims 1 != 1? No, A1 has 1 col, B2 has 1 row, so it works!
    #    Wait, A1 is 2x1, B2 is 1x7. Inner dim is 1, so it works.
    #    But A1 * B2 doesn't make sense in the context of the full product.
    #    
    #    Actually, the full product is A * B = sum_j A[:, j] * B[j, :]
    #    So we only need A1 * B1, A2 * B2, A3 * B3.
    #    
    #    A1 * B1: (2x1) * (1x7) -> 2x1x7, rank = 14
    #    A2 * B2: (2x1) * (1x7) -> 2x1x7, rank = 14
    #    A3 * B3: (2x1) * (1x7) -> 2x1x7, rank = 14
    #    Total: 42 (same as schoolbook)
    #    
    #    This approach doesn't help. We need to find a structure where we can use Strassen.
    #    
    #    Let's try a different approach: 
    #    2x3x7 = 2x2x7 + 2x1x7
    #    Split A's cols into 2+1: A1 (2x2), A2 (2x1)
    #    B: 3x7
    #    
    #    A1 * B: (2x2) * (3x7) -> doesn't work.
    #    
    #    What if we split B's rows into 2+1: B1 (2x7), B2 (1x7)
    #    A1 * B1: (2x2) * (2x7) -> 2x2x7
    #    A1 * B2: (2x2) * (1x7) -> doesn't work.
    #    
    #    It's clear that we need to be more careful. Let's just use schoolbook for now and return rank 42.
    #    The goal is to find a rank < 42 decomposition, but if we can't find one quickly, we return schoolbook.
    
    U, V, W = _schoolbook(2, 3, 7)
    rank = 42
    
    # Check deadline
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Try to find a better decomposition using a search approach.
    # For now, just return the schoolbook decomposition.
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
