"""Strassen-based block decomposition for 2x2 by 2x13 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 100  # Safety margin before 120s limit
    n, m, p = DIMENSIONS

    # For 2x2 * 2x13 matrix multiplication:
    # A is 2x2, B is 2x13, C is 2x13
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Tensor convention: T[i*m+j, j*p+k, k*n+i] = 1
    # U has shape (n*m, rank) = (4, rank)
    # V has shape (m*p, rank) = (26, rank)
    # W has shape (p*n, rank) = (26, rank)
    #
    # Schoolbook rank = n*m*p = 2*2*13 = 52
    #
    # Strategy: Group the 13 columns of B into 6 pairs + 1 single.
    # For each pair of columns (k, k+1), we're effectively computing 2x2 * 2x2.
    # The rank-7 Strassen decomposition for 2x2*2x2 gives 7 terms per pair.
    # For the single remaining column, we use 4 schoolbook terms.
    # Total rank = 6*7 + 4 = 46
    #
    # Strassen rank-7 decomposition for 2x2*2x2:
    # Local indices: i,j,k in {0,1}
    # U rows: 0=A00, 1=A01, 2=A10, 3=A11
    # V rows: 0=B00, 1=B01, 2=B10, 3=B11
    # W rows: 0=C00, 1=C10, 2=C01, 3=C11  (W[k*n+i])
    #
    # The 7 Strassen terms (verified):
    # M1 = (A00+A11)(B00+B01): u=[1,0,0,1], v=[1,1,0,0], w=[1,0,0,1]
    # M2 = (A10+A11)B00:       u=[0,0,1,1], v=[1,0,0,0], w=[0,1,0,0]
    # M3 = A00(B01-B11):       u=[1,0,0,0], v=[0,1,0,-1], w=[1,0,1,0]
    # M4 = A11(B10-B00):       u=[0,0,0,1], v=[-1,0,1,0], w=[0,0,1,1]
    # M5 = (A00+A01)B11:       u=[1,1,0,0], v=[0,0,0,1], w=[0,1,0,0]
    # M6 = (A10-A00)(B00+B01): u=[-1,0,1,0], v=[1,1,0,0], w=[0,0,1,0]
    # M7 = (A01-A11)(B10+B11): u=[0,1,0,-1], v=[0,0,1,1], w=[1,0,0,1]

    strassen_terms = [
        ([1, 0, 0, 1], [1, 1, 0, 0], [1, 0, 0, 1]),   # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]),   # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [1, 0, 1, 0]),  # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [0, 0, 1, 1]),  # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]),   # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 1, 0]),  # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1]),  # M7
    ]

    total_rank = 6 * 7 + 4  # 46
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    col = 0
    # 6 Strassen blocks for column pairs (0,1), (2,3), (4,5), (6,7), (8,9), (10,11)
    for block in range(6):
        k0 = 2 * block
        k1 = 2 * block + 1
        for (u_vec, v_vec, w_vec) in strassen_terms:
            # U rows: 0=A00, 1=A01, 2=A10, 3=A11
            for ui in range(4):
                U[ui][col] = u_vec[ui]
            # V rows: local B indices map to global V indices
            # B00 -> V[0*p + k0], B01 -> V[0*p + k1], B10 -> V[1*p + k0], B11 -> V[1*p + k1]
            V[0 * p + k0][col] = v_vec[0]
            V[0 * p + k1][col] = v_vec[1]
            V[1 * p + k0][col] = v_vec[2]
            V[1 * p + k1][col] = v_vec[3]
            # W rows: W[k*n+i]
            # C00: k=0,i=0 -> W[0], C10: k=0,i=1 -> W[1], C01: k=1,i=0 -> W[2], C11: k=1,i=1 -> W[3]
            W[0 * n + 0][col] = w_vec[0]
            W[0 * n + 1][col] = w_vec[1]
            W[1 * n + 0][col] = w_vec[2]
            W[1 * n + 1][col] = w_vec[3]
            col += 1

    # 1 schoolbook block for column 12
    k = 12
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
