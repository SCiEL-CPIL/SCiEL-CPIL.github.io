"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before the 150s wall-clock kill

    # Primary: rank-7 Strassen-based exact decomposition of M2x2 * M2x14.
    # The four A-rows are shared across all 14 output columns, so the 7
    # per-pair Strassen products collapse into 7 shared scalar products.
    result_7 = _construct_strassen_7(n, m, p)
    if result_7 is not None:
        return result_7

    # Fallback: schoolbook rank-56 (always valid).
    return _construct_schoolbook(n, m, p)


def _construct_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _construct_strassen_7(n, m, p):
    """
    Rank-7 exact decomposition of (n x m) * (m x p) with n=2, m=2, p=14.

    We treat C = A*B as a 2x14 matrix. Strassen's 7-product formula on the
    2x2 block structure gives 7 scalar products, each of the form
        prod_r = (linear combo of A-rows) . (linear combo of B-rows)
    where the A-side is a 2x2 matrix of coefficients (rows i, cols j) and the
    B-side is a 2x14 matrix of coefficients (rows j, cols k). The 7 products
    are then linearly combined (per output column k) to reconstruct C.

    U[i*m+j, r] = A-row coefficient   (4 x 7)
    V[j*p+k, r] = B-row coefficient   (28 x 7)
    W[k*n+i, r] = output coefficient  (28 x 7)
    """
    if n != 2 or m != 2:
        return None

    rank = 7
    U = [[0] * rank for _ in range(n * m)]   # 4 x 7
    V = [[0] * rank for _ in range(m * p)]   # 28 x 7
    W = [[0] * rank for _ in range(p * n)]   # 28 x 7

    # A-side coefficient matrices (rows i in {0,1}, cols j in {0,1}).
    # Standard Strassen 7 products:
    #   M1 = (A00 + A11) * (B0 + B1)
    #   M2 = (A10 + A11) * B0
    #   M3 =  A00        * (B1 - B0)
    #   M4 =  A11        * (B1 - B0)
    #   M5 = (A00 + A01) * B1
    #   M6 = (A10 - A00) * B0
    #   M7 = (A01 - A11) * B1
    # where B0 = row0 of B, B1 = row1 of B.
    A_coeffs = [
        [[1, 0], [1, 1]],   # M1: A00 + A11
        [[0, 0], [1, 1]],   # M2: A10 + A11
        [[1, 0], [0, 0]],   # M3: A00
        [[0, 0], [0, 1]],   # M4: A11
        [[1, 1], [0, 0]],   # M5: A00 + A01
        [[-1, 0], [1, 0]],  # M6: A10 - A00
        [[0, 1], [0, -1]],  # M7: A01 - A11
    ]

    # B-side coefficient vectors per row j (length p).
    #   row0 of B0:  M1:+1, M2:+1, M3:-1, M4:-1, M5:0,  M6:+1, M7:0
    #   row1 of B1:  M1:+1, M2:0,  M3:+1, M4:+1, M5:+1, M6:0,  M7:+1
    B_row0 = [1, 1, -1, -1, 0, 1, 0]
    B_row1 = [1, 0, 1, 1, 1, 0, 1]

    # Output (W) coefficient per column k: for output C[i,k], coefficient of M_r.
    #   C0 = M1 + M3 - M5
    #   C1 = M1 - M2 + M4 + M5 - M6 + M7
    out_row0 = [1, 0, 1, 0, -1, 0, 0]   # C0
    out_row1 = [1, -1, 0, 1, 1, -1, 1]  # C1

    for r in range(rank):
        # U: A-side coefficients
        U[0][r] = A_coeffs[r][0][0]
        U[1][r] = A_coeffs[r][0][1]
        U[2][r] = A_coeffs[r][1][0]
        U[3][r] = A_coeffs[r][1][1]

        # V: B-side coefficients (row j, column k)
        for k in range(p):
            V[0 * p + k][r] = B_row0[r]
            V[1 * p + k][r] = B_row1[r]

        # W: output coefficients (column k, row i)
        for k in range(p):
            W[k * n + 0][r] = out_row0[r]
            W[k * n + 1][r] = out_row1[r]

    if _verify_decomposition_fast(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    return None


def _verify_decomposition_fast(U, V, W, n, m, p, num_tests=5):
    """
    Verify the decomposition computes C = A @ B exactly.
    C[i,k] = sum_r W[k*n+i, r] * sum_j U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k]
    """
    rank = len(U[0])
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)

    U_r = U_np.reshape(n, m, rank)
    V_r = V_np.reshape(m, p, rank)
    W_r = W_np.reshape(p, n, rank)

    for _ in range(num_tests):
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B

        C_computed = np.zeros((n, p))
        for r in range(rank):
            A_U = A * U_r[:, :, r]      # (n, m)
            B_V = B * V_r[:, :, r]      # (m, p)
            term = A_U @ B_V            # (n, p)
            C_computed += term * W_r[:, :, r].T  # W_r.T is (n, p)

        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True
# EVOLVE-BLOCK-END
