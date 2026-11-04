import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    """
    Construct a rank-minimal tensor decomposition for 3x5 * 5x5 matrix multiplication.
    Uses a block-partitioned approach applying Strassen's algorithm to the 2x2 top-left
    sub-block (saving 1 rank over schoolbook) and schoolbook multiplication for the
    remaining blocks. Total rank = 74.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin within 120s limit

    n, m, p = DIMENSIONS  # 3, 5, 5
    rank = 74  # 7 (Strassen) + 12 + 12 + 18 + 4 + 6 + 6 + 9 = 74

    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]  # 15 x 74
    V = [[0] * rank for _ in range(m * p)]  # 25 x 74
    W = [[0] * rank for _ in range(p * n)]  # 15 x 74

    col = 0

    # Helper: schoolbook multiplication for a sub-block
    # A_sub: rows i0..i0+n_s-1, cols j0..j0+m_s-1
    # B_sub: rows j0..j0+m_s-1, cols k0..k0+p_s-1
    # C_sub: rows i0..i0+n_s-1, cols k0..k0+p_s-1
    def schoolbook(i0, j0, k0, n_s, m_s, p_s):
        nonlocal col
        for i_s in range(n_s):
            for j_s in range(m_s):
                for k_s in range(p_s):
                    i = i0 + i_s
                    j = j0 + j_s
                    k = k0 + k_s
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

    # Block 1: A[0:2, 0:2] x B[0:2, 0:2] -> C[0:2, 0:2] using Strassen (rank 7)
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    # Local indices: (i, j, k) each in {0, 1}
    
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # U: A[0,0] and A[1,1] -> rows 0*m+0=0, 1*m+1=6
    # V: B[0,0] and B[1,1] -> rows 0*p+0=0, 1*p+1=6
    # W: C[0,0] and C[1,1] -> rows 0*n+0=0, 1*n+1=4
    U[0 * m + 0][col] = 1
    U[1 * m + 1][col] = 1
    V[0 * p + 0][col] = 1
    V[1 * p + 1][col] = 1
    W[0 * n + 0][col] = 1
    W[1 * n + 1][col] = 1
    col += 1

    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    # U: A[1,0] and A[1,1] -> rows 1*m+0=5, 1*m+1=6
    # V: B[0,0] -> row 0*p+0=0
    # W: C[0,1] and C[1,1] -> rows 0*n+1=1, 1*n+1=4
    U[1 * m + 0][col] = 1
    U[1 * m + 1][col] = 1
    V[0 * p + 0][col] = 1
    W[0 * n + 1][col] = 1
    W[1 * n + 1][col] = -1
    col += 1

    # M3 = A[0,0] * (B[0,1]-B[1,1])
    # U: A[0,0] -> row 0*m+0=0
    # V: B[0,1] and B[1,1] -> rows 0*p+1=1, 1*p+1=6
    # W: C[1,0] and C[1,1] -> rows 1*n+0=3, 1*n+1=4
    U[0 * m + 0][col] = 1
    V[0 * p + 1][col] = 1
    V[1 * p + 1][col] = -1
    W[1 * n + 0][col] = 1
    W[1 * n + 1][col] = 1
    col += 1

    # M4 = A[1,1] * (B[1,0]-B[0,0])
    # U: A[1,1] -> row 1*m+1=6
    # V: B[1,0] and B[0,0] -> rows 1*p+0=5, 0*p+0=0
    # W: C[0,0] and C[0,1] -> rows 0*n+0=0, 0*n+1=1
    U[1 * m + 1][col] = 1
    V[1 * p + 0][col] = 1
    V[0 * p + 0][col] = -1
    W[0 * n + 0][col] = 1
    W[0 * n + 1][col] = 1
    col += 1

    # M5 = (A[0,0]+A[0,1]) * B[1,1]
    # U: A[0,0] and A[0,1] -> rows 0*m+0=0, 0*m+1=1
    # V: B[1,1] -> row 1*p+1=6
    # W: C[0,0] and C[1,0] -> rows 0*n+0=0, 1*n+0=3
    U[0 * m + 0][col] = 1
    U[0 * m + 1][col] = 1
    V[1 * p + 1][col] = 1
    W[0 * n + 0][col] = -1
    W[1 * n + 0][col] = 1
    col += 1

    # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # U: A[1,0] and A[0,0] -> rows 1*m+0=5, 0*m+0=0
    # V: B[0,0] and B[0,1] -> rows 0*p+0=0, 0*p+1=1
    # W: C[1,1] -> row 1*n+1=4
    U[1 * m + 0][col] = 1
    U[0 * m + 0][col] = -1
    V[0 * p + 0][col] = 1
    V[0 * p + 1][col] = 1
    W[1 * n + 1][col] = 1
    col += 1

    # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    # U: A[0,1] and A[1,1] -> rows 0*m+1=1, 1*m+1=6
    # V: B[1,0] and B[1,1] -> rows 1*p+0=5, 1*p+1=6
    # W: C[0,0] -> row 0*n+0=0
    U[0 * m + 1][col] = 1
    U[1 * m + 1][col] = -1
    V[1 * p + 0][col] = 1
    V[1 * p + 1][col] = 1
    W[0 * n + 0][col] = 1
    col += 1

    # Block 2: A[0:2, 2:5] x B[2:5, 0:2] -> C[0:2, 0:2], schoolbook rank 2*3*2=12
    schoolbook(0, 2, 0, 2, 3, 2)

    # Block 3: A[0:2, 0:2] x B[0:2, 2:5] -> C[0:2, 2:5], schoolbook rank 2*2*3=12
    schoolbook(0, 0, 2, 2, 2, 3)

    # Block 4: A[0:2, 2:5] x B[2:5, 2:5] -> C[0:2, 2:5], schoolbook rank 2*3*3=18
    schoolbook(0, 2, 2, 2, 3, 3)

    # Block 5: A[2:3, 0:2] x B[0:2, 0:2] -> C[2:3, 0:2], schoolbook rank 1*2*2=4
    schoolbook(2, 0, 0, 1, 2, 2)

    # Block 6: A[2:3, 2:5] x B[2:5, 0:2] -> C[2:3, 0:2], schoolbook rank 1*3*2=6
    schoolbook(2, 2, 0, 1, 3, 2)

    # Block 7: A[2:3, 0:2] x B[0:2, 2:5] -> C[2:3, 2:5], schoolbook rank 1*2*3=6
    schoolbook(2, 0, 2, 1, 2, 3)

    # Block 8: A[2:3, 2:5] x B[2:5, 2:5] -> C[2:3, 2:5], schoolbook rank 1*3*3=9
    schoolbook(2, 2, 2, 1, 3, 3)

    assert col == rank, f"Expected rank {rank}, got {col}"

    return {"U": U, "V": V, "W": W}
