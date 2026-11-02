"""Rank-minimal exact tensor decomposition for 4x4 * 4x9 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]
_START = time.time()
_DEADLINE = 138.0  # safety margin below the 150s kill

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """T[io, u, v] = 1 iff output (i,k) uses a[i,j]*b[j,k]; io=k*n+i, u=i*m+j, v=j*p+k."""
    T = np.zeros((p * n, n * m, m * p), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k * n + i, i * m + j, j * p + k] = 1
    return T


def _exact_check(U, V, W, T, n, m, p):
    rank = len(U[0])
    Ua = np.array(U, dtype=object)
    Va = np.array(V, dtype=object)
    Wa = np.array(W, dtype=object)
    for io in range(p * n):
        for u in range(n * m):
            for v in range(m * p):
                s = 0
                for c in range(rank):
                    s += Ua[u, c] * Va[v, c] * Wa[io, c]
                if s != T[io, u, v]:
                    return False
    return True


def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
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
    return {"U": U, "V": V, "W": W}


def _strassen_2x2x2():
    """
    Strassen decomposition for 2x2 * 2x2.
    Returns U (4x7), V (4x7), W (4x7) where:
    U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c] sums to 1 iff (i,j,k) is a valid triple.
    
    Strassen formulas:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B12-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    rank = 7
    U = [[0] * rank for _ in range(4)]  # 2*2 = 4
    V = [[0] * rank for _ in range(4)]  # 2*2 = 4
    W = [[0] * rank for _ in range(4)]  # 2*2 = 4
    
    # Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    # i,j,k in {0,1}
    
    # M1 = (A11+A22)(B11+B22)
    # A11 = u[0], A22 = u[3]
    # B11 = v[0], B22 = v[3]
    # C11 = w[0], C22 = w[3]
    U[0][0] = 1  # A11
    U[3][0] = 1  # A22
    V[0][0] = 1  # B11
    V[3][0] = 1  # B22
    W[0][0] = 1  # C11
    W[3][0] = 1  # C22
    
    # M2 = (A21+A22)B11
    # A21 = u[2], A22 = u[3]
    # B11 = v[0]
    # C21 = w[2]
    U[2][1] = 1
    U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3 = A11(B21-B22)
    # A11 = u[0]
    # B21 = v[2], B22 = v[3]
    # C12 = w[1], C22 = w[3]
    U[0][2] = 1
    V[2][2] = 1
    V[3][2] = -1
    W[1][2] = 1
    W[3][2] = 1
    
    # M4 = A22(B12-B11)
    # A22 = u[3]
    # B12 = v[1], B11 = v[0]
    # C11 = w[0], C21 = w[2]
    U[3][3] = 1
    V[1][3] = 1
    V[0][3] = -1
    W[0][3] = 1
    W[2][3] = 1
    
    # M5 = (A11+A12)B22
    # A11 = u[0], A12 = u[1]
    # B22 = v[3]
    # C11 = w[0] (with -1), C12 = w[1]
    U[0][4] = 1
    U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1
    W[1][4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    # A21 = u[2], A11 = u[0]
    # B11 = v[0], B12 = v[1]
    # C22 = w[3]
    U[2][5] = 1
    U[0][5] = -1
    V[0][5] = 1
    V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    # A12 = u[1], A22 = u[3]
    # B21 = v[2], B22 = v[3]
    # C11 = w[0], C22 = w[3]
    U[1][6] = 1
    U[3][6] = -1
    V[2][6] = 1
    V[3][6] = 1
    W[0][6] = 1
    W[3][6] = 1
    
    return U, V, W


def _strassen_2x2x4():
    """
    Decomposition for 2x2 * 2x4 using Strassen-like approach.
    Split B into two 2x2 blocks: B = [B_left, B_right]
    C = [C_left, C_right] where C_left = A*B_left, C_right = A*B_right
    
    Use Strassen for each 2x2*2x2, total rank 14.
    U (4x14), V (8x14), W (8x14)
    """
    rank = 14
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(8)]
    W = [[0] * rank for _ in range(8)]
    
    U1, V1, W1 = _strassen_2x2x2()
    U2, V2, W2 = _strassen_2x2x2()
    
    # For B_left: V indices 0-3, W indices 0-3
    for i in range(4):
        for c in range(7):
            U[i][c] += U1[i][c]
            V[i][c] += V1[i][c]
            W[i][c] += W1[i][c]
    
    # For B_right: V indices 4-7, W indices 4-7
    for i in range(4):
        for c in range(7):
            U[i][c + 7] += U2[i][c]
            V[i + 4][c + 7] += V2[i][c]
            W[i + 4][c + 7] += W2[i][c]
    
    return U, V, W


def _strassen_2x2x9():
    """
    Decomposition for 2x2 * 2x9.
    Split B into four 2x2 blocks and one 2x1 block.
    Total rank: 4*7 + 4 = 32 (schoolbook for 2x2x1 is 4)
    """
    rank = 4 * 7 + 4  # 32
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(18)]  # 2*9 = 18
    W = [[0] * rank for _ in range(18)]  # 2*9 = 18
    
    U1, V1, W1 = _strassen_2x2x2()
    
    # Block 0: cols 0-1
    for i in range(4):
        for c in range(7):
            U[i][c] += U1[i][c]
            V[i][c] += V1[i][c]
            W[i][c] += W1[i][c]
    
    # Block 1: cols 2-3
    for i in range(4):
        for c in range(7):
            U[i][c + 7] += U1[i][c]
            V[i + 2][c + 7] += V1[i][c]
            W[i + 2][c + 7] += W1[i][c]
    
    # Block 2: cols 4-5
    for i in range(4):
        for c in range(7):
            U[i][c + 14] += U1[i][c]
            V[i + 4][c + 14] += V1[i][c]
            W[i + 4][c + 14] += W1[i][c]
    
    # Block 3: cols 6-7
    for i in range(4):
        for c in range(7):
            U[i][c + 21] += U1[i][c]
            V[i + 6][c + 21] += V1[i][c]
            W[i + 6][c + 21] += W1[i][c]
    
    # Block 4: col 8 (2x1, schoolbook rank 4)
    for i in range(4):
        U[i][i + 28] = 1
        V[i + 8][i + 28] = 1
        W[i + 8][i + 28] = 1
    
    return U, V, W


def _decompose_4x4x9():
    """
    Decompose 4x4 * 4x9 using block structure.
    A = [[A11, A12], [A21, A22]] where each Ak1 is 2x2.
    B is 4x9, partitioned by rows: B = [B_top; B_bottom] where B_top, B_bottom are 2x9.
    C = A*B:
    C[0:2, :] = A11*B_top + A12*B_bottom
    C[2:4, :] = A21*B_top + A22*B_bottom
    
    Each term is 2x2 * 2x9. Use _strassen_2x2x9() for each.
    Total rank: 4 * 32 = 128.
    """
    n, m, p = 4, 4, 9
    rank = 4 * 32  # 128
    U = [[0] * rank for _ in range(16)]  # 4*4
    V = [[0] * rank for _ in range(36)]  # 4*9
    W = [[0] * rank for _ in range(36)]  # 9*4
    
    U1, V1, W1 = _strassen_2x2x9()
    
    # A11 * B_top: A11 is rows 0-1, cols 0-1 of A; B_top is rows 0-1 of B
    # U indices: i*4+j for i,j in 0..1 -> 0..3
    # V indices: j*9+k for j in 0..1, k in 0..8 -> 0..17
    # W indices: k*4+i for k in 0..8, i in 0..1 -> 0..35 (i=0,1)
    for i in range(4):
        for c in range(32):
            U[i][c] += U1[i][c]
            V[i][c] += V1[i][c]
            W[i][c] += W1[i][c]
    
    # A12 * B_bottom: A12 is rows 0-1, cols 2-3 of A; B_bottom is rows 2-3 of B
    # U indices: i*4+(j+2) for i in 0..1, j in 0..1 -> 2,3,6,7
    # V indices: (j+2)*9+k for j in 0..1, k in 0..8 -> 18..35
    # W indices: k*4+i for k in 0..8, i in 0..1 -> 0..35 (i=0,1)
    for i in range(4):
        for c in range(32):
            U[i + 2][c + 32] += U1[i][c]
            V[i + 18][c + 32] += V1[i][c]
            W[i][c + 32] += W1[i][c]
    
    # A21 * B_top: A21 is rows 2-3, cols 0-1 of A; B_top is rows 0-1 of B
    # U indices: (i+2)*4+j for i in 0..1, j in 0..1 -> 8,9,12,13
    # V indices: j*9+k for j in 0..1, k in 0..8 -> 0..17
    # W indices: k*4+(i+2) for k in 0..8, i in 0..1 -> 16..35 (i=2,3)
    for i in range(4):
        for c in range(32):
            U[i + 8][c + 64] += U1[i][c]
            V[i][c + 64] += V1[i][c]
            W[i + 16][c + 64] += W1[i][c]
    
    # A22 * B_bottom: A22 is rows 2-3, cols 2-3 of A; B_bottom is rows 2-3 of B
    # U indices: (i+2)*4+(j+2) for i in 0..1, j in 0..1 -> 10,11,14,15
    # V indices: (j+2)*9+k for j in 0..1, k in 0..8 -> 18..35
    # W indices: k*4+(i+2) for k in 0..8, i in 0..1 -> 16..35 (i=2,3)
    for i in range(4):
        for c in range(32):
            U[i + 10][c + 96] += U1[i][c]
            V[i + 18][c + 96] += V1[i][c]
            W[i + 16][c + 96] += W1[i][c]
    
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    # Try Strassen-based decomposition (rank 128)
    best = _decompose_4x4x9()
    if _exact_check(best["U"], best["V"], best["W"], T, n, m, p):
        return best
    
    # Fallback to schoolbook (rank 144)
    best = _schoolbook(n, m, p)
    if _exact_check(best["U"], best["V"], best["W"], T, n, m, p):
        return best
    
    # Should not reach here
    return best

# EVOLVE-BLOCK-END
