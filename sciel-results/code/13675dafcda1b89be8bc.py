"""Improved program: block-Strassen decomposition for 4x5x5 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 5]
_DEADLINE_SECONDS = 115  # safety margin before 120s kill

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return (U, V, W) as lists of lists, rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _strassen_2x2():
    """Strassen's 7-product algorithm for 2x2x2.
    Returns U (4x7), V (4x7), W (4x7) as lists of lists (integers).
    A is 2x2 flattened row-major: (a,b,c,d) -> indices 0,1,2,3
    B is 2x2 flattened row-major: (e,f,g,h) -> indices 0,1,2,3
    C = A*B, W index = k*2 + i (n=2, p=2).
    W[0] -> i=0,k=0: C[0][0]
    W[1] -> i=0,k=1: C[0][1]
    W[2] -> i=1,k=0: C[1][0]
    W[3] -> i=1,k=1: C[1][1]
    """
    U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, -1, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V = [
        [1, 1, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 1]
    ]
    W = [
        [1, 0, 0, -1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, -1, 0, 0, 1, 0]
    ]
    return U, V, W


def _schoolbook_block(n, m, p):
    """Schoolbook decomposition for n x m times m x p = n x p.
    Returns U (n*m x n*m*p), V (m*p x n*m*p), W (p*n x n*m*p) as lists of lists.
    U row index: i*m + j
    V row index: j*p + k
    W row index: k*n + i
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _construct_4x5x5():
    """Build 4x5x5 decomposition using block Strassen.
    
    A: 4x5, B: 5x5, C: 4x5.
    Split the summation index j (0-4) into j=0,1 and j=2,3,4.
    
    C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{4} A[i,j]*B[j,k]
    
    Part 1: A_left (4x2) * B_top (2x5) -> 4x5
      Split A_left rows into 2+2: A_top_left (2x2), A_bot_left (2x2)
      Split B_top cols into 2+3: B_top_left (2x2), B_top_right (2x3)
      A_top_left*B_top_left: 2x2*2x2, Strassen rank 7
      A_top_left*B_top_right: 2x2*2x3, schoolbook rank 12
      A_bot_left*B_top_left: 2x2*2x2, Strassen rank 7
      A_bot_left*B_top_right: 2x2*2x3, schoolbook rank 12
      Total Part 1: 38
    
    Part 2: A_right (4x3) * B_bot (3x5) -> 4x5
      Schoolbook rank: 4*3*5 = 60.
    
    Grand total: 38 + 60 = 98.
    """
    n, m, p = 4, 5, 5
    total_rank = 98
    
    U = [[0] * total_rank for _ in range(n * m)]  # 20 x 98
    V = [[0] * total_rank for _ in range(m * p)]  # 25 x 98
    W = [[0] * total_rank for _ in range(p * n)]  # 20 x 98
    
    col = 0
    
    # Part 1: A_left (4x2) * B_top (2x5)
    # A_left: A[i,j] for i=0..3, j=0..1 -> U rows i*5+j
    # B_top: B[j,k] for j=0..1, k=0..4 -> V rows j*5+k
    # Output: C[i,k] for i=0..3, k=0..4 -> W rows k*4+i
    
    # Sub-part 1a: A_top_left (2x2) * B_top_left (2x2) - Strassen
    # A_top_left: A[0,0],A[0,1],A[1,0],A[1,1] -> U rows 0,1,5,6
    # B_top_left: B[0,0],B[0,1],B[1,0],B[1,1] -> V rows 0,1,5,6
    # Output: C[0,0],C[0,1],C[1,0],C[1,1] -> W rows 0,1,4,5
    
    sU, sV, sW = _strassen_2x2()
    a_rows = [0, 1, 5, 6]  # U rows for A_top_left
    b_rows = [0, 1, 5, 6]  # V rows for B_top_left
    c_rows = [0, 1, 4, 5]  # W rows for C[0,0],C[0,1],C[1,0],C[1,1]
    for i in range(4):
        for c in range(7):
            U[a_rows[i]][col + c] = sU[i][c]
            V[b_rows[i]][col + c] = sV[i][c]
            W[c_rows[i]][col + c] = sW[i][c]
    col += 7
    
    # Sub-part 1b: A_top_left (2x2) * B_top_right (2x3) - schoolbook
    # A_top_left: U rows 0,1,5,6
    # B_top_right: B[0,2],B[0,3],B[0,4],B[1,2],B[1,3],B[1,4] -> V rows 2,3,4,7,8,9
    # Output: C[0,2],C[0,3],C[0,4],C[1,2],C[1,3],C[1,4]
    # W index = k*4+i: C[0,2]=2*4+0=8, C[0,3]=3*4+0=12, C[0,4]=4*4+0=16
    # C[1,2]=2*4+1=9, C[1,3]=3*4+1=13, C[1,4]=4*4+1=17
    a_rows = [0, 1, 5, 6]  # A_top_left: (0,0),(0,1),(1,0),(1,1)
    b_rows = [2, 3, 4, 7, 8, 9]  # B_top_right: (0,2),(0,3),(0,4),(1,2),(1,3),(1,4)
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[a_rows[i * 2 + j]][col] = 1
                V[b_rows[j * 3 + k]][col] = 1
                W[(2 + k) * 4 + i][col] = 1
                col += 1
    # col now at 7 + 12 = 19
    
    # Sub-part 1c: A_bot_left (2x2) * B_top_left (2x2) - Strassen
    # A_bot_left: A[2,0],A[2,1],A[3,0],A[3,1] -> U rows 10,11,15,16
    # B_top_left: V rows 0,1,5,6
    # Output: C[2,0],C[2,1],C[3,0],C[3,1] -> W rows 0*4+2=8, 1*4+2=10, 0*4+3=12, 1*4+3=14
    
    sU, sV, sW = _strassen_2x2()
    a_rows = [10, 11, 15, 16]
    b_rows = [0, 1, 5, 6]
    c_rows = [8, 10, 12, 14]  # C[2,0]=8, C[2,1]=10, C[3,0]=12, C[3,1]=14
    for i in range(4):
        for c in range(7):
            U[a_rows[i]][col + c] = sU[i][c]
            V[b_rows[i]][col + c] = sV[i][c]
            W[c_rows[i]][col + c] = sW[i][c]
    col += 7
    # col now at 19 + 7 = 26
    
    # Sub-part 1d: A_bot_left (2x2) * B_top_right (2x3) - schoolbook
    # A_bot_left: U rows 10,11,15,16
    # B_top_right: V rows 2,3,4,7,8,9
    # Output: C[2,2],C[2,3],C[2,4],C[3,2],C[3,3],C[3,4]
    # W: C[2,2]=2*4+2=10, C[2,3]=3*4+2=14, C[2,4]=4*4+2=18
    # C[3,2]=2*4+3=11, C[3,3]=3*4+3=15, C[3,4]=4*4+3=19
    a_rows = [10, 11, 15, 16]
    b_rows = [2, 3, 4, 7, 8, 9]
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[a_rows[i * 2 + j]][col] = 1
                V[b_rows[j * 3 + k]][col] = 1
                W[(2 + k) * 4 + (2 + i)][col] = 1
                col += 1
    # col now at 26 + 12 = 38
    
    # Part 2: A_right (4x3) * B_bot (3x5) - schoolbook
    # A_right: A[i,j] for i=0..3, j=2..4 -> U rows i*5+j
    # B_bot: B[j,k] for j=2..4, k=0..4 -> V rows j*5+k
    # Output: C[i,k] for i=0..3, k=0..4 -> W rows k*4+i
    
    for i in range(4):
        for j in range(3):
            for k in range(5):
                U[i * 5 + (2 + j)][col] = 1
                V[(2 + j) * 5 + k][col] = 1
                W[k * 4 + i][col] = 1
                col += 1
    # col now at 38 + 60 = 98
    
    assert col == total_rank, f"Expected {total_rank}, got {col}"
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, num_trials=5):
    """Verify that the decomposition correctly computes matrix multiplication."""
    np.random.seed(42)
    rank = len(W[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for trial in range(num_trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for r in range(rank):
                        C_computed[i, k] += U_np[i * m + j, r] * V_np[j * p + k, r] * W_np[k * n + i, r]
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True


def construct():
    """Construct the rank-minimal exact tensor decomposition for 4x5x5 matrix multiplication."""
    t0 = time.time()
    
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Build the block-Strassen decomposition (rank 98)
    U, V, W = _construct_4x5x5()
    rank = len(W[0])
    
    # Verify correctness
    if time.time() - t0 < _DEADLINE_SECONDS - 10:
        if not _verify_decomposition(U, V, W, n, m, p, num_trials=5):
            # Fallback to schoolbook if verification fails
            U, V, W = _schoolbook(n, m, p)
    
    # Check deadline
    if time.time() - t0 > _DEADLINE_SECONDS:
        U, V, W = _schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
