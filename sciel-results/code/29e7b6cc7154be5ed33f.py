"""Tensor decomposition for 2x3 x 3x8 matrix multiplication."""
import time

DIMENSIONS = [2, 3, 8]


def construct():
    n, m, p = DIMENSIONS
    rank = 44  # 28 (Strassen for 2x2 blocks) + 16 (scalar for 3rd row)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # W index convention: W[k*n + i, r]
    # U index: row = i*m + j  (i in [0,1], j in [0,1,2])
    # V index: row = j*p + k  (j in [0,1,2], k in [0,7])

    col = 0
    # Part 1: Strassen on A_0,1 (rows 0,1) x B_0,1 (rows 0,1), columns 0..7
    # We have 4 pairs of columns: (0,1), (2,3), (4,5), (6,7)
    # For each pair k=0..3 (representing columns 2k, 2k+1):
    for k in range(4):
        c0 = 2 * k
        c1 = 2 * k + 1
        # Strassen 7 products for 2x2 x 2x2:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)(B00)
        # M3 = (A00)(B10 + B11)
        # M4 = (A00 - A10)(B01 + B10)
        # M5 = (A01 - A11)(B11 + B00)
        # M6 = (A10 - A00)(B11)
        # M7 = (A11 - A01)(B01)
        #
        # Here A is 2x2: A00=A[0,0], A01=A[0,1], A10=A[1,0], A11=A[1,1]
        # B is 2x2: B00=B[0,c0], B01=B[0,c1], B10=B[1,c0], B11=B[1,c1]
        # C00 = M1 - M2 + M3 - M5
        # C01 = M2 - M4 - M7
        # C10 = M5 + M6
        # C11 = M1 - M3 + M4 + M7
        #
        # U rows: A00=0, A01=1, A10=3, A11=4
        # V rows: B[0,c0]=c0, B[0,c1]=c1, B[1,c0]=8+c0, B[1,c1]=8+c1
        # W rows: C[0,c0]=c0*2+0=2*c0, C[0,c1]=2*c1, C[1,c0]=2*c0+1, C[1,c1]=2*c1+1

        # M1 = (A00 + A11)(B00 + B11)
        U[0][col] = 1  # A00
        U[4][col] = 1  # A11
        V[c0][col] = 1    # B00
        V[8 + c1][col] = 1  # B11
        W[2 * c0][col] = 1     # C00
        W[2 * c1][col] = -1    # C01
        W[2 * c0 + 1][col] = -1  # C10
        W[2 * c1 + 1][col] = 1   # C11
        col += 1

        # M2 = (A10 + A11)(B00)
        U[3][col] = 1  # A10
        U[4][col] = 1  # A11
        V[c0][col] = 1  # B00
        W[2 * c0][col] = -1    # C00
        W[2 * c1][col] = 1     # C01
        W[2 * c0 + 1][col] = 0  # C10
        W[2 * c1 + 1][col] = 0  # C11
        col += 1

        # M3 = (A00)(B10 + B11)
        U[0][col] = 1  # A00
        V[8 + c0][col] = 1  # B10
        V[8 + c1][col] = 1  # B11
        W[2 * c0][col] = 1   # C00
        W[2 * c1][col] = 0   # C01
        W[2 * c0 + 1][col] = 0  # C10
        W[2 * c1 + 1][col] = 0  # C11
        col += 1

        # M4 = (A00 - A10)(B01 + B10)
        U[0][col] = 1  # A00
        U[3][col] = -1  # -A10
        V[c1][col] = 1  # B01
        V[8 + c0][col] = 1  # B10
        W[2 * c0][col] = 0   # C00
        W[2 * c1][col] = -1  # C01
        W[2 * c0 + 1][col] = 0  # C10
        W[2 * c1 + 1][col] = 1   # C11
        col += 1

        # M5 = (A01 - A11)(B11 + B00)
        U[1][col] = 1  # A01
        U[4][col] = -1  # -A11
        V[8 + c1][col] = 1  # B11
        V[c0][col] = 1  # B00
        W[2 * c0][col] = -1   # C00
        W[2 * c1][col] = 0   # C01
        W[2 * c0 + 1][col] = 1  # C10
        W[2 * c1 + 1][col] = 0  # C11
        col += 1

        # M6 = (A10 - A00)(B11)
        U[3][col] = 1  # A10
        U[0][col] = -1  # -A00
        V[8 + c1][col] = 1  # B11
        W[2 * c0][col] = 0   # C00
        W[2 * c1][col] = 0   # C01
        W[2 * c0 + 1][col] = 1  # C10
        W[2 * c1 + 1][col] = 0  # C11
        col += 1

        # M7 = (A11 - A01)(B01)
        U[4][col] = 1  # A11
        U[1][col] = -1  # -A01
        V[c1][col] = 1  # B01
        W[2 * c0][col] = 0   # C00
        W[2 * c1][col] = -1  # C01
        W[2 * c0 + 1][col] = 0  # C10
        W[2 * c1 + 1][col] = 1   # C11
        col += 1

    # Part 2: Scalar multiplications for A column 2 x B row 2
    # A[0,2] * B[2,k] for k=0..7 -> C[0,k]
    # A[1,2] * B[2,k] for k=0..7 -> C[1,k]
    for k in range(8):
        # C[0,k] = A[0,2] * B[2,k]
        U[2][col] = 1   # A[0,2], index 0*3+2=2
        V[16 + k][col] = 1  # B[2,k], index 2*8+k=16+k
        W[2 * k][col] = 1  # C[0,k], index k*2+0=2k
        col += 1

        # C[1,k] = A[1,2] * B[2,k]
        U[5][col] = 1   # A[1,2], index 1*3+2=5
        V[16 + k][col] = 1  # B[2,k], index 2*8+k=16+k
        W[2 * k + 1][col] = 1  # C[1,k], index k*2+1=2k+1
        col += 1

    return {"U": U, "V": V, "W": W}
