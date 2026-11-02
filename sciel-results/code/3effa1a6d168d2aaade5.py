"""Rewritten program for 2x2 * 2x6 matrix multiplication tensor decomposition."""
import time

DIMENSIONS = [2, 2, 6]


def _strassen_2x2x2():
    """
    Return (U, V, W) for 2x2 * 2x2 -> 2x2 using Strassen's 7-product algorithm.
    U shape (4,7), V shape (4,7), W shape (4,7).
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c] for i,j,k in 0..1.
    """
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]

    # Strassen products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # Each Strassen product c uses linear combinations of A entries (i,j) and B entries (j,k).
    # U[i*2+j, c] = coefficient of A[i,j] in product c
    # V[j*2+k, c] = coefficient of B[j,k] in product c
    # W[k*2+i, c] = coefficient of product c in C[i,k]

    # Product 0: M1 = (A00+A11)(B00+B11)
    # U: A00=1, A11=1; V: B00=1, B11=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    # C00 += M1, C11 += M1
    W[0][0] = 1; W[3][1] = 1

    # Product 1: M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # C10 += M2, C11 -= M2
    W[0][1] = 1; W[3][0] = -1

    # Product 2: M3 = A00(B10-B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    # C01 += M3, C11 += M3
    W[2][0] = 1; W[3][1] = 1

    # Product 3: M4 = A11(B01-B00)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    # C00 += M4, C10 += M4
    W[1][0] = 1; W[0][1] = 1

    # Product 4: M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    # C00 -= M5, C01 += M5
    W[3][0] = -1; W[2][0] = 1

    # Product 5: M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # C11 += M6
    W[3][1] = 1

    # Product 6: M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    # C00 += M7
    W[0][0] = 1

    return U, V, W


def construct():
    """
    Construct a tensor decomposition for 2x2 * 2x6 -> 2x6.
    Strategy: Split the 2x6 matrix B into three 2x2 blocks along the k dimension.
    Use Strassen's algorithm (rank 7) for each 2x2x2 block multiplication.
    Total rank = 3 * 7 = 21.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=6

    # We have 3 blocks of size 2x2 in the k dimension.
    # Block b handles k indices 2*b and 2*b+1.
    num_blocks = p // m  # 3 blocks

    U, V, W = _strassen_2x2x2()

    # Build the full decomposition.
    # U has shape (n*m, rank) = (4, 21)
    # V has shape (m*p, rank) = (12, 21)
    # W has shape (p*n, rank) = (12, 21)
    rank = num_blocks * 7

    U_full = [[0] * rank for _ in range(n * m)]
    V_full = [[0] * rank for _ in range(m * p)]
    W_full = [[0] * rank for _ in range(p * n)]

    for b in range(num_blocks):
        base_col = b * 7
        for c in range(7):
            col = base_col + c
            # U: same for all blocks (A is shared)
            for i in range(n):
                for j in range(m):
                    U_full[i * m + j][col] = U[i * m + j][c]

            # V: B block b corresponds to B[j, 2*b + k]
            for j in range(m):
                for k in range(m):
                    # V index: j*p + (2*b + k)
                    V_full[j * p + (2 * b + k)][col] = V[j * m + k][c]

            # W: C block b corresponds to C[2*b + k, i]
            for k in range(m):
                for i in range(n):
                    # W index: (2*b + k)*n + i
                    W_full[(2 * b + k) * n + i][col] = W[k * n + i][c]

    return {"U": U_full, "V": V_full, "W": W_full}
