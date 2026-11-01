"""Matrix multiplication 2x2x2 — rank-7 Strassen decomposition."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the rank-7 Strassen decomposition for 2x2 matrix multiplication.

    U: (4, 7) — shape (n*m, rank)
    V: (4, 7) — shape (m*p, rank)
    W: (4, 7) — shape (p*n, rank), following the trace convention W[k*n+i, r]

    The decomposition is exact: for any 2x2 matrices A, B with C = A*B,
    C_{i,k} = sum_r U[i*r + k, r] * V[k*r + j, r] * W[j*n + i, r]
    (indices adjusted to the W[k*n+i, r] convention).

    Strassen's algorithm uses 7 multiplications instead of the 8 needed
    by the schoolbook method. Rank 7 is provably minimal for 2x2x2
    matrix multiplication over fields of characteristic not 2 (Landsberg
    et al.), so this is the theoretical optimum.
    """
    # Strassen's 7 intermediate products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # Then:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # U[i,j] indexes A (i in 0..3 for A00,A01,A10,A11; j in 0..6)
    # V[j,i] indexes B (same ordering for B00,B01,B10,B11; j in 0..6)
    # W[k,i] follows the trace convention W[k*n+i, r] for output C[k,i]
    #   k is the output row, i is the output column, r is the factor index
    #   So W is indexed as W[0..3, 0..6] where row = k*2 + i

    U = [
        # A00 (row 0): coefficients for M1..M7
        [1, 0, 1, 0, 1, -1, 0],
        # A01 (row 1)
        [0, 0, 0, 0, 1, 0, 1],
        # A10 (row 2)
        [0, 1, 0, 0, 0, 1, 0],
        # A11 (row 3)
        [1, 1, 0, 1, 0, 0, -1]
    ]

    V = [
        # B00 (row 0)
        [1, 1, 0, -1, 0, 1, 0],
        # B01 (row 1)
        [0, 0, 1, 0, 0, 1, 0],
        # B10 (row 2)
        [0, 0, 0, 1, 0, 0, 1],
        # B11 (row 3)
        [1, 0, -1, 0, 1, 0, 1]
    ]

    # W follows the convention W[k*n + i, r] where k=output row, i=output col, r=factor
    # C00 = M1 + M4 - M5 + M7  -> row 0 (k=0, i=0)
    # C01 = M3 + M5            -> row 1 (k=0, i=1)
    # C10 = M2 + M4            -> row 2 (k=1, i=0)
    # C11 = M1 - M2 + M3 + M6  -> row 3 (k=1, i=1)
    W = [
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0]   # C11
    ]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
