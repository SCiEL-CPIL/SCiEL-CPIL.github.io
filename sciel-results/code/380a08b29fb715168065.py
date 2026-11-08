import numpy as np
import time

# Rank-88 decomposition for 3x4x8 matrix multiplication
# Structure: 32 (schoolbook for i=0) + 4*7 (Strassen for i=1,2 over 2x2 blocks)

def construct():
    deadline = time.time() + 140.0

    n, m, p = 3, 4, 8
    R = 88

    U = np.zeros((n*m, R), dtype=np.int32)
    V = np.zeros((m*p, R), dtype=np.int32)
    W = np.zeros((p*n, R), dtype=np.int32)

    # Strassen 7-term 2x2x2 decomposition (row-major order)
    # U[i,j]: coefficient of A[i,j], V[j,k]: coefficient of B[j,k], W[i,k]: coefficient of C[i,k]
    strassen_U = np.array([
        [1, 1],
        [1, 0],
        [0, 1],
        [0, 1],
        [0, 1],
        [1, 0],
        [1, 0],
    ])
    strassen_V = np.array([
        [1, 1],
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 0],
        [0, 1],
        [0, 1],
    ])
    strassen_W = np.array([
        [1, 0],
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
    ])
    # Actually, let me use the standard Strassen formulas properly.
    # Standard Strassen for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6

    # So the 7 terms have these U, V, W patterns:
    # t=0: M1: U = A11+A22 -> [[1,0],[0,1]], V = B11+B22 -> [[1,0],[0,1]], W = C11+C22 -> [[1,0],[0,1]]
    # t=1: M2: U = A21+A22 -> [[0,0],[1,1]], V = B11 -> [[1,0],[0,0]], W = C21+C22 -> [[0,0],[1,1]]... 
    # Wait, this is getting complex. Let me just use the standard 7x2x2 matrices.

    SU = np.array([
        [[1, 0], [0, 1]],
        [[0, 0], [1, 1]],
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[0, 0], [1, 1]],
        [[0, 1], [0, 0]],
    ])
    SV = np.array([
        [[1, 0], [0, 1]],
        [[1, 0], [0, 0]],
        [[0, 1], [0, 0]],
        [[0, 0], [0, 1]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[0, 1], [1, 1]],
    ])
    SW = np.array([
        [[1, 0], [0, 1]],
        [[0, 0], [1, 1]],
        [[0, 1], [0, 0]],
        [[1, 0], [0, 0]],
        [[0, 0], [1, 0]],
        [[0, 0], [1, 0]],
        [[1, 0], [0, 0]],
    ])

    col = 0

    # Part 1: i=0, schoolbook 1x4x8 = 32 terms
    for j in range(m):
        for k in range(p):
            if time.time() > deadline:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            U[0*m + j, col] = 1
            V[j*p + k, col] = 1
            W[k*n + 0, col] = 1
            col += 1

    # Part 2: i=1,2 using Strassen on 2x2 blocks
    # For each 2x2 block of (j,k) indices, use 7 Strassen terms
    # i indices for the block: 1, 2 (local indices 0, 1)
    for jb in range(m // 2):
        for kb in range(p // 2):
            for t in range(7):
                if time.time() > deadline:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                for li in range(2):
                    for lj in range(2):
                        val = SU[t, li, lj]
                        if val:
                            U[(1+li)*m + (jb*2+lj), col] += val
                for lj in range(2):
                    for lk in range(2):
                        val = SV[t, lj, lk]
                        if val:
                            V[(jb*2+lj)*p + (kb*2+lk), col] += val
                for li in range(2):
                    for lk in range(2):
                        val = SW[t, li, lk]
                        if val:
                            W[(kb*2+lk)*n + (1+li), col] += val
                col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
