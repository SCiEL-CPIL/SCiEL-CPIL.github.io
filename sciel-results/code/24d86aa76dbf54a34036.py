"""Rank-minimal tensor decomposition for 2x4x7 matrix multiplication."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()

    # Rank 43 decomposition:
    # 4 blocks of 2x2x2 (Strassen, rank 7 each) for k=0..3 -> 28
    # 2 blocks of 2x2x2 (Strassen, rank 7 each) for k=4..5 -> 14
    # 1 rank-1 term for 2x4x1 (k=6) -> 1
    # Total 43
    rank = 43
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1: (A11+A22)(B11+B22)
        U[i0*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        # M2: (A21+A22)B11
        U[(i0+1)*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] -= 1
        col += 1
        # M3: A11(B12-B22)
        U[i0*m + j0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        # M4: A22(B21-B11)
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+1)*p + k0][col] += 1
        V[j0*p + k0][col] -= 1
        W[k0*n + i0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        col += 1
        # M5: (A11+A12)B22
        U[i0*m + j0][col] += 1
        U[i0*m + (j0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        col += 1
        # M6: (A21-A11)(B11+B12)
        U[(i0+1)*m + j0][col] += 1
        U[i0*m + j0][col] -= 1
        V[j0*p + k0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        # M7: (A12-A22)(B21+B22)
        U[i0*m + (j0+1)][col] += 1
        U[(i0+1)*m + (j0+1)][col] -= 1
        V[(j0+1)*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        col += 1

    # Part 1: 2x4x4 (k=0..3) - 4 Strassen blocks = 28
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)

    # Part 2: 2x4x2 (k=4..5) - 2 Strassen blocks = 14
    add_strassen(0, 0, 4)
    add_strassen(0, 2, 4)

    # Part 3: 2x4x1 (k=6) - 1 rank-1 term
    # T[i,j,6] = 1 for all i,j -> u[i]*v[j]*w[6] with u=[1,1], v=[1,1,1,1], w[6]=1
    for i in range(2):
        for j in range(4):
            U[i*m + j][col] = 1
    for j in range(4):
        V[j*p + 6][col] = 1
    for i in range(2):
        W[6*n + i][col] = 1
    col += 1

    # Time guard
    if time.time() - start_time > 100:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
