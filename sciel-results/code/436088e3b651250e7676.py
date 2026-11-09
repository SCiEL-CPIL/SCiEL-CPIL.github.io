"""Optimized matrix multiplication tensor decomposition for 6x6."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 10s safety margin

    n, m, p = DIMENSIONS

    # Strategy: Block-wise Strassen decomposition
    # 6x6 = 3x3 blocks of 2x2
    # Each 2x2 multiplication uses Strassen (rank 7)
    # Total: 3*3*3 block triples * 7 Strassen terms = 189
    #
    # This is a valid exact decomposition. We verify it and return it.
    #
    # The key insight from the inspirations: the Strassen 2x2 decomposition
    # is well-defined and can be tiled across the 3x3 block grid.
    #
    # For each (i, j, k) block triple (i=row block, j=middle block, k=col block):
    #   A[i,j] (2x2) * B[j,k] (2x2) contributes to C[i,k] (2x2)
    #   Using Strassen: 7 scalar multiplications
    #
    # Total rank = 27 * 7 = 189

    rank = 189

    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    # Strassen 2x2 multiplication terms:
    # For A (2x2), B (2x2) -> C (2x2)
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    #
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5

    # Define Strassen factors as lists of (local_row, local_col, coeff)
    # For U: A indices (row, col)
    # For V: B indices (row, col)
    # For W: C indices (row, col)

    strassen_U = [
        [(0, 0, 1), (1, 1, 1)],       # M0: A00 + A11
        [(1, 0, 1), (1, 1, 1)],       # M1: A10 + A11
        [(0, 0, 1)],                   # M2: A00
        [(1, 1, 1)],                   # M3: A11
        [(0, 0, 1), (0, 1, 1)],       # M4: A00 + A01
        [(1, 0, 1), (0, 0, -1)],      # M5: A10 - A00
        [(0, 1, 1), (1, 1, -1)],      # M6: A01 - A11
    ]

    strassen_V = [
        [(0, 0, 1), (1, 1, 1)],       # M0: B00 + B11
        [(0, 0, 1)],                   # M1: B00
        [(0, 1, 1), (1, 1, -1)],      # M2: B01 - B11
        [(1, 0, 1), (0, 0, -1)],      # M3: B10 - B00
        [(1, 1, 1)],                   # M4: B11
        [(0, 0, 1), (0, 1, 1)],       # M5: B00 + B01
        [(1, 0, 1), (1, 1, 1)],       # M6: B10 + B11
    ]

    strassen_W = [
        [(0, 0, 1), (1, 1, 1)],       # M0: C00 + C11
        [(1, 0, 1), (1, 1, -1)],      # M1: C10 - C11
        [(0, 1, 1), (1, 1, 1)],       # M2: C01 + C11
        [(0, 0, 1), (1, 0, 1)],       # M3: C00 + C10
        [(0, 0, -1), (0, 1, 1)],      # M4: -C00 + C01
        [(1, 1, 1)],                   # M5: C11
        [(0, 0, 1)],                   # M6: C00
    ]

    # Build the decomposition by tiling Strassen across the 3x3 block grid
    col = 0
    for i in range(3):      # row block of A/C
        for j in range(3):  # middle block (col of A, row of B)
            for k in range(3):  # col block of B/C
                for t in range(7):  # Strassen term
                    # U: A entries
                    # A is indexed as [row * m + col] in the U matrix
                    # Local A position (lr, lc) -> global (i*2+lr, j*2+lc)
                    for lr, lc, coeff in strassen_U[t]:
                        u_row = (i * 2 + lr) * m + (j * 2 + lc)
                        U[u_row, col] = coeff

                    # V: B entries
                    # B is indexed as [row * p + col] in the V matrix
                    # Local B position (lr, lc) -> global (j*2+lr, k*2+lc)
                    for lr, lc, coeff in strassen_V[t]:
                        v_row = (j * 2 + lr) * p + (k * 2 + lc)
                        V[v_row, col] = coeff

                    # W: C entries
                    # W uses trace convention: W[k*n + i, r]
                    # where i is row of C, k is col of C
                    # So W index = col_of_C * n + row_of_C
                    # Local C position (lr, lc) -> global row (i*2+lr), col (k*2+lc)
                    for lr, lc, coeff in strassen_W[t]:
                        w_row = (k * 2 + lc) * n + (i * 2 + lr)
                        W[w_row, col] = coeff

                    col += 1

    # Verify the decomposition is valid
    # Check: sum_r U[:,r] * V[:,r] * W[:,r] should equal the multiplication tensor
    # The multiplication tensor T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(i,k) for some i,j,k
    # i.e., T[i*m+j, j*p+k, k*n+i] = 1 (with appropriate index mapping)
    #
    # Actually, let's verify by checking a random case
    # C = A @ B should equal sum_r (U[:,r] dot A_flat) * (V[:,r] dot B_flat) * W[:,r]
    # where A_flat = A.reshape(-1), B_flat = B.reshape(-1)
    # and result = W.T @ (element-wise product)

    # Quick verification with a random matrix
    rng = np.random.default_rng(42)
    A_test = rng.standard_normal((n, m))
    B_test = rng.standard_normal((m, p))
    C_expected = A_test @ B_test

    # Compute via decomposition
    A_flat = A_test.reshape(-1)
    B_flat = B_test.reshape(-1)
    # For each rank factor r: scalar_r = (U[:,r] . A_flat) * (V[:,r] . B_flat)
    # C_flat[r] = scalar_r * W[:,r]
    # C_flat = sum_r scalar_r * W[:,r]
    scalars = (U.T @ A_flat) * (V.T @ B_flat)
    C_flat_computed = W.T @ scalars
    C_computed = C_flat_computed.reshape((n, p))

    # Check if close
    err = np.max(np.abs(C_computed - C_expected))
    if err > 1e-10:
        # Fallback: use schoolbook (rank 216)
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)

        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1

    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
