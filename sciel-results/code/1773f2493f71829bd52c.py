import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-28 exact tensor decomposition for 2x4 * 4x4 matrix multiplication.
    
    Strategy: Split the middle dimension (size 4) into two halves of size 2.
    A (2x4) = [A1 (2x2) | A2 (2x2)]
    B (4x4) = [B11 (2x2) B12 (2x2); B21 (2x2) B22 (2x2)]
    
    C = A * B = [A1*B11 + A2*B21 | A1*B12 + A2*B22]
    
    Each 2x2 * 2x2 product is computed via Strassen (rank 7).
    Four such blocks give total rank 4*7 = 28.
    
    Index conventions:
      U[i*m + j, r]  — coefficient for A[i,j]
      V[j*p + k, r]  — coefficient for B[j,k]
      W[k*n + i, r]  — coefficient for C[i,k]  (trace convention: k*n+i)
    """
    n, m, p = DIMENSIONS  # 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]   # 8 x 28
    V = [[0] * rank for _ in range(m * p)]   # 16 x 28
    W = [[0] * rank for _ in range(p * n)]   # 8 x 28
    col = 0

    def add_strassen_block(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        """
        Add a rank-7 Strassen decomposition for a 2x2*2x2 sub-problem.
        a_rows: [r0, r1] rows of A used
        a_cols: [c0, c1] cols of A used
        b_rows: [r0, r1] rows of B used
        b_cols: [c0, c1] cols of B used
        c_rows: [r0, r1] rows of C (output)
        c_cols: [c0, c1] cols of C (output)
        """
        nonlocal col
        # Global indices
        u00 = a_rows[0] * m + a_cols[0]
        u01 = a_rows[0] * m + a_cols[1]
        u10 = a_rows[1] * m + a_cols[0]
        u11 = a_rows[1] * m + a_cols[1]

        v00 = b_rows[0] * p + b_cols[0]
        v01 = b_rows[0] * p + b_cols[1]
        v10 = b_rows[1] * p + b_cols[0]
        v11 = b_rows[1] * p + b_cols[1]

        w00 = c_cols[0] * n + c_rows[0]
        w01 = c_cols[1] * n + c_rows[0]
        w10 = c_cols[0] * n + c_rows[1]
        w11 = c_cols[1] * n + c_rows[1]

        # M1 = (P00+P11)*(Q00+Q11)
        U[u00][col] += 1; U[u11][col] += 1
        V[v00][col] += 1; V[v11][col] += 1
        W[w00][col] += 1; W[w11][col] += 1
        col += 1
        # M2 = (P10+P11)*Q00
        U[u10][col] += 1; U[u11][col] += 1
        V[v00][col] += 1
        W[w10][col] += 1; W[w11][col] += -1
        col += 1
        # M3 = P00*(Q01-Q11)
        U[u00][col] += 1
        V[v01][col] += 1; V[v11][col] += -1
        W[w01][col] += 1; W[w11][col] += 1
        col += 1
        # M4 = P11*(Q10-Q00)
        U[u11][col] += 1
        V[v10][col] += 1; V[v00][col] += -1
        W[w00][col] += 1; W[w10][col] += 1
        col += 1
        # M5 = (P00+P01)*Q11
        U[u00][col] += 1; U[u01][col] += 1
        V[v11][col] += 1
        W[w01][col] += 1; W[w00][col] += -1
        col += 1
        # M6 = (P10-P00)*(Q00+Q01)
        U[u10][col] += 1; U[u00][col] += -1
        V[v00][col] += 1; V[v01][col] += 1
        W[w11][col] += 1
        col += 1
        # M7 = (P01-P11)*(Q10+Q11)
        U[u01][col] += 1; U[u11][col] += -1
        V[v10][col] += 1; V[v11][col] += 1
        W[w00][col] += 1
        col += 1

    # Block 1: A[:,0:2] * B[0:2,0:2] -> C[:,0:2]
    add_strassen_block([0,1], [0,1], [0,1], [0,1], [0,1], [0,1])
    # Block 2: A[:,2:4] * B[2:4,0:2] -> C[:,0:2]
    add_strassen_block([0,1], [2,3], [2,3], [0,1], [0,1], [0,1])
    # Block 3: A[:,0:2] * B[0:2,2:4] -> C[:,2:4]
    add_strassen_block([0,1], [0,1], [0,1], [2,3], [0,1], [2,3])
    # Block 4: A[:,2:4] * B[2:4,2:4] -> C[:,2:4]
    add_strassen_block([0,1], [2,3], [2,3], [2,3], [0,1], [2,3])

    assert col == rank
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
