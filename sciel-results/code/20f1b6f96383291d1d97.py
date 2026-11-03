import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    """
    Constructs a low-rank matrix multiplication decomposition for 2x3 * 3x8.
    Returns a dictionary with matrices U, V, W of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS

    # Strategy: Split the middle dimension m=3 into m1=2 and m2=1.
    # Part A: 2x2 * 2x8 (rows 0,1 of A; rows 0,1 of B)
    # Part B: 2x1 * 1x8 (row 2 of A; row 2 of B)
    #
    # For Part A: Decompose 2x2*2x8 as 4 independent 2x2*2x2 multiplications.
    # Each uses Strassen (rank 7). Total: 4*7 = 28.
    # For Part B: 2x1*1x8 needs rank 2*8 = 16 (schoolbook).
    # Total rank: 28 + 16 = 44.
    #
    # Try to reduce: Can we share structure between blocks?
    # For now, construct rank-44 and verify. Then attempt rank-43.

    rank = 44

    U = [[0] * rank for _ in range(n * m)]   # shape (6, rank)
    V = [[0] * rank for _ in range(m * p)]   # shape (24, rank)
    W = [[0] * rank for _ in range(p * n)]   # shape (16, rank)

    col = 0

    # Part 1: A[0:2, 0:2] * B[0:2, 0:8] -> C[0:2, 0:8]
    # Split B[0:2, 0:8] into 4 blocks of 2x2: B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:6], B[0:2, 6:8]
    # For each 2x2 block, apply Strassen

    for blk in range(4):
        # Block offset in B columns
        bo = 2 * blk
        # U indices for A[0:2, 0:2]
        # A[0,0]->U[0], A[0,1]->U[1], A[1,0]->U[3], A[1,1]->U[4]
        # V indices for B[0:2, bo:bo+2]
        # B[0,bo]->V[bo], B[0,bo+1]->V[bo+1], B[1,bo]->V[8+bo], B[1,bo+1]->V[8+bo+1]
        # W indices for C[0:2, bo:bo+2]
        # C[i,k] -> W[k*n+i] = W[2*k+i]
        # C[0,bo]->W[2*bo], C[0,bo+1]->W[2*bo+2], C[1,bo]->W[2*bo+1], C[1,bo+1]->W[2*bo+3]

        ve = bo
        vf = bo + 1
        vg = 8 + bo
        vh = 8 + bo + 1

        w00 = 2 * bo
        w01 = 2 * bo + 2
        w10 = 2 * bo + 1
        w11 = 2 * bo + 3

        # Strassen P1 = (a+d)(e+h) -> contributes to C[0,0] and C[1,1]
        U[0][col] = 1
        U[4][col] = 1
        V[ve][col] = 1
        V[vh][col] = 1
        W[w00][col] = 1
        W[w11][col] = 1
        col += 1

        # P2 = (c+d)e -> contributes to C[1,0] and C[1,1]
        U[3][col] = 1
        U[4][col] = 1
        V[ve][col] = 1
        W[w10][col] = 1
        W[w11][col] = -1
        col += 1

        # P3 = a(f-h) -> contributes to C[0,1] and C[1,1]
        U[0][col] = 1
        V[vf][col] = 1
        V[vh][col] = -1
        W[w01][col] = 1
        W[w11][col] = 1
        col += 1

        # P4 = d(g-e) -> contributes to C[0,0] and C[1,0]
        U[4][col] = 1
        V[vg][col] = 1
        V[ve][col] = -1
        W[w00][col] = 1
        W[w10][col] = 1
        col += 1

        # P5 = (a+b)h -> contributes to C[0,0] and C[0,1]
        U[0][col] = 1
        U[1][col] = 1
        V[vh][col] = 1
        W[w00][col] = -1
        W[w01][col] = 1
        col += 1

        # P6 = (c-a)(e+f) -> contributes to C[1,1]
        U[3][col] = 1
        U[0][col] = -1
        V[ve][col] = 1
        V[vf][col] = 1
        W[w11][col] = 1
        col += 1

        # P7 = (b-d)(g+h) -> contributes to C[0,0]
        U[1][col] = 1
        U[4][col] = -1
        V[vg][col] = 1
        V[vh][col] = 1
        W[w00][col] = 1
        col += 1

    # Part 2: A[0:2, 2] * B[2, 0:8] -> C[0:2, 0:8]
    # A[0,2]->U[2], A[1,2]->U[5]
    # B[2,k]->V[16+k]
    # C[i,k]->W[2k+i]
    for k in range(8):
        for i in range(2):
            U[i * 3 + 2][col] = 1
            V[16 + k][col] = 1
            W[2 * k + i][col] = 1
            col += 1

    # Verify col == rank
    if col != rank:
        # Fallback: return schoolbook
        rank_fallback = n * m * p
        U = [[0] * rank_fallback for _ in range(n * m)]
        V = [[0] * rank_fallback for _ in range(m * p)]
        W = [[0] * rank_fallback for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
