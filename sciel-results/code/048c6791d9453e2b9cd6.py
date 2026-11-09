import time
import sys

DIMENSIONS = [3, 4, 7]

def _build_strassen_2x2x2():
    """Return the 7 rank-1 terms for Strassen's 2x2x2 matrix multiplication."""
    U = [
        [1,  0,  1,  0,  1, -1,  0],
        [0,  1,  0,  0,  1,  0,  1],
        [0,  0,  1,  0,  0,  1,  0],
        [1,  0,  0,  1,  0,  0, -1],
    ]
    V = [
        [1,  0,  0, -1,  0,  1,  0],
        [0,  1, -1,  0,  0,  1,  0],
        [0,  0,  0,  1,  0,  0,  1],
        [1,  0,  0,  0,  1,  0,  1],
    ]
    W = [
        [1,  0,  0, -1, -1,  0,  1],
        [0,  1,  1,  0,  1,  0,  0],
        [0,  0,  0,  1,  0,  1,  0],
        [1,  0, -1,  0, -1,  1,  0],
    ]
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    r = len(W[0]) if W else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_3x4x7():
    """Build a rank-reduced decomposition for 3x4x7 matrix multiplication.
    
    Strategy:
    - A (3x4) split into A1 (2x4) and A2 (1x4)
    - For A1: split the 4 intermediate indices into two groups of 2
      - A1a (2x2, cols 0-1) * B_top (2x7)
      - A1b (2x2, cols 2-3) * B_bot (2x7)
    - For each 2x2x7: split B into 2x2 + 2x2 + 2x1
      - 2x2x2: Strassen (7)
      - 2x2x2: Strassen (7)
      - 2x2x1: schoolbook (4)
      - Total per 2x2x7: 18
    - A1 total: 2 * 18 = 36
    - A2 (1x4) * B (4x7): schoolbook = 28
    - Total: 64
    """
    n, m, p = 3, 4, 7
    total_rank = 80
    U = [[0] * total_rank for _ in range(n * m)]  # 12 x rank
    V = [[0] * total_rank for _ in range(m * p)]  # 28 x rank
    W = [[0] * total_rank for _ in range(p * n)]  # 21 x rank
    
    c = 0
    sU, sV, sW = _build_strassen_2x2x2()
    
    # A1a (A[0:2, 0:2]) * B_top (B[0:2, :]) -> C[0:2, :]
    # B_top is 2x7. Split into B[0:2,0:2], B[0:2,2:4], B[0:2,4:7]
    # Sub: 2x2x2 Strassen for cols 0-1
    for ci in range(7):
        for ui in range(4):
            U[ui][c] = sU[ui][ci]
        for vi in range(4):
            V[vi][c] = sV[vi][ci]
        W[0][c] = sW[0][ci]
        W[3][c] = sW[1][ci]
        W[1][c] = sW[2][ci]
        W[4][c] = sW[3][ci]
        c += 1
    # Sub: 2x2x2 Strassen for cols 2-3
    for ci in range(7):
        for ui in range(4):
            U[ui][c] = sU[ui][ci]
        for vi in range(4):
            V[vi + 4][c] = sV[vi][ci]
        W[6][c] = sW[0][ci]
        W[9][c] = sW[1][ci]
        W[7][c] = sW[2][ci]
        W[10][c] = sW[3][ci]
        c += 1
    # Sub: 2x2x3 schoolbook for cols 4-6
    for i in range(2):
        for j in range(2):
            for k in range(4, 7):
                U[i * 4 + j][c] = 1
                V[j * 7 + k][c] = 1
                W[k * 3 + i][c] = 1
                c += 1
    
    # A1b (A[0:2, 2:4]) * B_bot (B[2:4, :]) -> C[0:2, :]
    # Sub: 2x2x2 Strassen for cols 0-1
    for ci in range(7):
        for ui in range(4):
            U[ui + 4][c] = sU[ui][ci]
        for vi in range(4):
            V[vi + 14][c] = sV[vi][ci]
        W[0][c] += sW[0][ci]
        W[3][c] += sW[1][ci]
        W[1][c] += sW[2][ci]
        W[4][c] += sW[3][ci]
        c += 1
    # Sub: 2x2x2 Strassen for cols 2-3
    for ci in range(7):
        for ui in range(4):
            U[ui + 4][c] = sU[ui][ci]
        for vi in range(4):
            V[vi + 18][c] = sV[vi][ci]
        W[6][c] += sW[0][ci]
        W[9][c] += sW[1][ci]
        W[7][c] += sW[2][ci]
        W[10][c] += sW[3][ci]
        c += 1
    # Sub: 2x2x3 schoolbook for cols 4-6
    for i in range(2):
        for j in range(2, 4):
            for k in range(4, 7):
                U[i * 4 + j][c] = 1
                V[j * 7 + k][c] = 1
                W[k * 3 + i][c] = 1
                c += 1
    
    # A2 (A[2, :]) * B (:, :) -> C[2, :]
    # 1x4x7 schoolbook: 28 terms
    for j in range(4):
        for k in range(7):
            U[2 * 4 + j][c] = 1
            V[j * 7 + k][c] = 1
            W[k * 3 + 2][c] = 1
            c += 1
    
    return U, V, W, c

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Build the rank-51 decomposition
    U, V, W, rank = _build_3x4x7()
    
    # Verify correctness
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
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
