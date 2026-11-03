"""Rank-44 exact tensor decomposition for 3x4 x 4x4 matrix multiplication."""
import time

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0


def construct():
    """
    Rank-44 exact tensor decomposition for 3x4 x 4x4 matrix multiplication.
    
    Strategy:
    - Split A (3x4) into top 2 rows and bottom 1 row.
    - Top 2 rows (2x4) x B (4x4): split into 4 blocks of 2x2 x 2x2,
      each using Strassen (rank 7). Total: 28.
    - Bottom 1 row (1x4) x B (4x4): split into 4 blocks of 1x2 x 2x2,
      each using schoolbook (rank 4). Total: 16.
    - Grand total rank: 44.
    """
    n, m, p = DIMENSIONS  # 3, 4, 4
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 12 x 44
    V = [[0] * rank for _ in range(m * p)]  # 16 x 44
    W = [[0] * rank for _ in range(p * n)]  # 12 x 44
    
    col = 0
    
    def add_strassen_2x2x2(a_idx, b_idx, c_idx):
        """
        Add Strassen 2x2x2 decomposition (rank 7).
        a_idx: list of 4 flat indices for A's 2x2 block [a00, a01, a10, a11]
        b_idx: list of 4 flat indices for B's 2x2 block [b00, b01, b10, b11]
        c_idx: list of 4 flat indices for C's 2x2 block [c00, c01, c10, c11]
        """
        nonlocal col
        a00, a01, a10, a11 = a_idx
        b00, b01, b10, b11 = b_idx
        c00, c01, c10, c11 = c_idx
        
        # m1 = (a00+a01)*(b00+b10)
        U[a00][col] = 1; U[a01][col] = 1
        V[b00][col] = 1; V[b10][col] = 1
        W[c00][col] = 1; W[c11][col] = 1
        col += 1
        
        # m2 = (a00+a01)*(b11-b10)
        U[a00][col] = 1; U[a01][col] = 1
        V[b11][col] = 1; V[b10][col] = -1
        W[c01][col] = 1; W[c10][col] = -1
        col += 1
        
        # m3 = (a10+a11)*b00
        U[a10][col] = 1; U[a11][col] = 1
        V[b00][col] = 1
        W[c10][col] = 1; W[c11][col] = -1
        col += 1
        
        # m4 = (a10+a11)*b11
        U[a10][col] = 1; U[a11][col] = 1
        V[b11][col] = 1
        W[c00][col] = 1; W[c11][col] = -1
        col += 1
        
        # m5 = a00*b01
        U[a00][col] = 1
        V[b01][col] = 1
        W[c01][col] = 1; W[c11][col] = 1
        col += 1
        
        # m6 = (a00-a10)*(b00+b10)
        U[a00][col] = 1; U[a10][col] = -1
        V[b00][col] = 1; V[b10][col] = 1
        W[c00][col] = -1; W[c10][col] = 1
        col += 1
        
        # m7 = (a01-a11)*(b11-b10)
        U[a01][col] = 1; U[a11][col] = -1
        V[b11][col] = 1; V[b10][col] = -1
        W[c01][col] = 1; W[c11][col] = -1
        col += 1
        
        return col
    
    def add_schoolbook_1x2x2(a_idx, b_idx, c_idx):
        """
        Add schoolbook 1x2 x 2x2 decomposition (rank 4).
        a_idx: list of 2 flat indices for A's 1x2 block [a0, a1]
        b_idx: list of 4 flat indices for B's 2x2 block [b00, b01, b10, b11]
        c_idx: list of 2 flat indices for C's 1x2 block [c0, c1]
        """
        nonlocal col
        a0, a1 = a_idx
        b00, b01, b10, b11 = b_idx
        c0, c1 = c_idx
        
        # Term 1: a0 * b00 -> c0
        U[a0][col] = 1
        V[b00][col] = 1
        W[c0][col] = 1
        col += 1
        
        # Term 2: a1 * b10 -> c0
        U[a1][col] = 1
        V[b10][col] = 1
        W[c0][col] = 1
        col += 1
        
        # Term 3: a0 * b01 -> c1
        U[a0][col] = 1
        V[b01][col] = 1
        W[c1][col] = 1
        col += 1
        
        # Term 4: a1 * b11 -> c1
        U[a1][col] = 1
        V[b11][col] = 1
        W[c1][col] = 1
        col += 1
        
        return col
    
    # Top 2 rows of A (rows 0,1) x B (4x4)
    # A top is 2x4, split columns into two 2x2 blocks
    # B is 4x4, split rows into two 2x2 blocks
    # This gives 4 sub-problems of 2x2 x 2x2
    
    # Sub-problem 1: A[0:2, 0:2] x B[0:2, 0:4] -> C[0:2, 0:2]
    a_idx_1 = [0, 1, 4, 5]
    b_idx_1 = [0, 1, 4, 5]
    c_idx_1 = [0, 1, 3, 4]
    col = add_strassen_2x2x2(a_idx_1, b_idx_1, c_idx_1)
    
    # Sub-problem 2: A[0:2, 0:2] x B[2:4, 0:4] -> C[0:2, 2:4]
    a_idx_2 = [0, 1, 4, 5]
    b_idx_2 = [8, 9, 12, 13]
    c_idx_2 = [6, 9, 7, 10]
    col = add_strassen_2x2x2(a_idx_2, b_idx_2, c_idx_2)
    
    # Sub-problem 3: A[0:2, 2:4] x B[0:2, 0:4] -> C[0:2, 0:2]
    a_idx_3 = [2, 3, 6, 7]
    b_idx_3 = [0, 1, 4, 5]
    c_idx_3 = [0, 1, 3, 4]
    col = add_strassen_2x2x2(a_idx_3, b_idx_3, c_idx_3)
    
    # Sub-problem 4: A[0:2, 2:4] x B[2:4, 0:4] -> C[0:2, 2:4]
    a_idx_4 = [2, 3, 6, 7]
    b_idx_4 = [8, 9, 12, 13]
    c_idx_4 = [6, 9, 7, 10]
    col = add_strassen_2x2x2(a_idx_4, b_idx_4, c_idx_4)
    
    # Bottom 1 row of A (row 2) x B (4x4)
    # A[2,0]=8, A[2,1]=9, A[2,2]=10, A[2,3]=11
    # Split into two 1x2 blocks
    
    # Block 1: A[2,0:2] x B[0:2, 0:4] -> C[2, 0:2]
    a_idx_5 = [8, 9]
    b_idx_5 = [0, 1, 4, 5]
    c_idx_5 = [6, 7]
    col = add_schoolbook_1x2x2(a_idx_5, b_idx_5, c_idx_5)
    
    # Block 2: A[2,0:2] x B[2:4, 0:4] -> C[2, 2:4]
    a_idx_6 = [8, 9]
    b_idx_6 = [8, 9, 12, 13]
    c_idx_6 = [8, 11]
    col = add_schoolbook_1x2x2(a_idx_6, b_idx_6, c_idx_6)
    
    # Block 3: A[2,2:4] x B[0:2, 0:4] -> C[2, 0:2]
    a_idx_7 = [10, 11]
    b_idx_7 = [0, 1, 4, 5]
    c_idx_7 = [6, 7]
    col = add_schoolbook_1x2x2(a_idx_7, b_idx_7, c_idx_7)
    
    # Block 4: A[2,2:4] x B[2:4, 0:4] -> C[2, 2:4]
    a_idx_8 = [10, 11]
    b_idx_8 = [8, 9, 12, 13]
    c_idx_8 = [8, 11]
    col = add_schoolbook_1x2x2(a_idx_8, b_idx_8, c_idx_8)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
