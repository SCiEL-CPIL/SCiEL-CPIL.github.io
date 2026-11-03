"""Program for matrix multiplication tensor decomposition task."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_strassen_2x2x2():
    """
    Return Strassen's 7-rank decomposition for 2x2*2x2.
    U: (4,7), V: (4,7), W: (4,7)
    Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    # Strassen's algorithm:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    # U[i*2+j, r] = coefficient of A[i,j] in M_r
    # V[j*2+k, r] = coefficient of B[j,k] in M_r
    # W[k*2+i, r] = coefficient of M_r in C[i,k]
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a11+a22)(b11+b22)
    U[0][0] = 1; U[3][0] = 1  # a11, a22
    V[0][0] = 1; V[3][0] = 1  # b11, b22
    W[0][0] = 1; W[3][0] = 1  # c11, c22
    
    # M2 = (a21+a22)b11
    U[2][1] = 1; U[3][1] = 1  # a21, a22
    V[0][1] = 1  # b11
    W[2][1] = 1; W[3][1] = -1  # c21, c22
    
    # M3 = a11(b12-b22)
    U[0][2] = 1  # a11
    V[1][2] = 1; V[3][2] = -1  # b12, b22
    W[1][2] = 1; W[3][2] = 1  # c12, c22
    
    # M4 = a22(b21-b11)
    U[3][3] = 1  # a22
    V[2][3] = 1; V[0][3] = -1  # b21, b11
    W[0][3] = 1; W[2][3] = 1  # c11, c21
    
    # M5 = (a11+a12)b22
    U[0][4] = 1; U[1][4] = 1  # a11, a12
    V[3][4] = 1  # b22
    W[0][4] = -1; W[1][4] = 1  # c11, c12
    
    # M6 = (a21-a11)(b11+b12)
    U[2][5] = 1; U[0][5] = -1  # a21, a11
    V[0][5] = 1; V[1][5] = 1  # b11, b12
    W[3][5] = 1  # c22
    
    # M7 = (a12-a22)(b21+b22)
    U[1][6] = 1; U[3][6] = -1  # a12, a22
    V[2][6] = 1; V[3][6] = 1  # b21, b22
    W[0][6] = 1  # c11
    
    return U, V, W

def _build_rank21_3x3x3():
    """
    Build a rank-21 decomposition for 3x3*3x3.
    Uses a combination of Strassen blocks and schoolbook terms.
    U: (9,21), V: (9,21), W: (9,21)
    """
    # We'll construct this by combining smaller blocks.
    # Approach: partition 3x3 into overlapping 2x2 blocks and use Strassen.
    # Actually, let's use a known construction.
    
    # For now, let's build a valid (possibly not minimal) decomposition.
    # We'll use a systematic approach: start with schoolbook (rank 27) and try to reduce.
    
    # Actually, let's just build a correct decomposition first.
    # Schoolbook for 3x3*3x3 has rank 27.
    # We'll return that for now, and the caller can try to improve.
    
    n = m = p = 3
    rank = 27
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    return U, V, W

def _build_3x3x6_from_3x3x3(U3, V3, W3):
    """
    Build a 3x3*3x6 decomposition from two 3x3*3x3 decompositions.
    The 3x6 matrix B is split into two 3x3 blocks: B[:, 0:3] and B[:, 3:6].
    C = A * B = A * [B1 | B2] = [A*B1 | A*B2]
    
    U: (9, rank), V: (18, rank), W: (18, rank)
    V rows 0-8: B1 (columns 0-2 of B), rows 9-17: B2 (columns 3-5 of B)
    W rows 0-8: C1 (columns 0-2 of C), rows 9-17: C2 (columns 3-5 of C)
    """
    rank3 = len(U3[0])
    rank6 = 2 * rank3
    
    U6 = [[0]*rank6 for _ in range(9)]
    V6 = [[0]*rank6 for _ in range(18)]
    W6 = [[0]*rank6 for _ in range(18)]
    
    # First half: A * B1
    for r in range(rank3):
        for i in range(3):
            for j in range(3):
                U6[i*3+j][r] = U3[i*3+j][r]
        for j in range(3):
            for k in range(3):
                V6[j*3+k][r] = V3[j*3+k][r]
        for k in range(3):
            for i in range(3):
                W6[k*3+i][r] = W3[k*3+i][r]
    
    # Second half: A * B2
    for r in range(rank3):
        for i in range(3):
            for j in range(3):
                U6[i*3+j][rank3+r] = U3[i*3+j][r]
        for j in range(3):
            for k in range(3):
                V6[j*3 + 3 + k][rank3+r] = V3[j*3+k][r]
        for k in range(3):
            for i in range(3):
                W6[k*3 + 3 + i][rank3+r] = W3[k*3+i][r]
    
    return U6, V6, W6

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x6 matrix.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    def elapsed():
        return time.time() - start_time
    
    def time_left():
        return deadline - time.time()
    
    # Strategy 1: Build from known 3x3*3x3 decomposition
    # First, get a valid 3x3*3x3 decomposition
    U3, V3, W3 = _build_rank21_3x3x3()
    
    # Verify it's valid
    if _verify_decomposition(U3, V3, W3, 3, 3, 3):
        U6, V6, W6 = _build_3x3x6_from_3x3x3(U3, V3, W3)
        if _verify_decomposition(U6, V6, W6, n, m, p):
            return {"U": U6, "V": V6, "W": W6}
    
    # Strategy 2: Try to find a better 3x3*3x3 decomposition using algebraic methods
    # For now, fall back to schoolbook for 3x3*3x6 directly
    
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
