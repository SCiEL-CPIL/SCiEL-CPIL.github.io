"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # A (2x4) split into 2 blocks of 2x2 along cols.
    # B (4x6) split into 2 blocks of 2x6 along rows (corresponding to A's col split).
    # Each 2x6 block is split into 3 blocks of 2x2 along cols.
    # Total 2x2x2 multiplications: 2 (from A split) * 3 (from B col split) = 6.
    # Rank of 2x2x2 is 7 (Strassen).
    # Total rank = 6 * 7 = 42.
    rank = 42
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # Local indices:
    # A: (0,0), (0,1), (1,0), (1,1)
    # B: (0,0), (0,1), (1,0), (1,1)
    # C: (0,0), (0,1), (1,0), (1,1)
    
    # Define the 7 Strassen terms as dictionaries mapping local (row, col) to coefficient
    # Term 0: P1 = (A00 + A11) * (B00 + B11)
    # Term 1: P2 = (A10 + A11) * B00
    # Term 2: P3 = A00 * (B01 - B11)
    # Term 3: P4 = A11 * (B10 - B00)
    # Term 4: P5 = (A00 + A01) * B11
    # Term 5: P6 = (A10 - A00) * (B00 + B01)
    # Term 6: P7 = (A01 - A11) * (B10 + B11)
    
    # U_coeffs[term][i][j] = coeff
    # V_coeffs[term][j][k] = coeff
    # W_coeffs[term][i][k] = coeff
    
    U_coeffs = [
        [[1, 0], [0, 1]],  # P1: A00, A11
        [[0, 0], [1, 1]],  # P2: A10, A11
        [[1, 0], [0, 0]],  # P3: A00
        [[0, 0], [0, 1]],  # P4: A11
        [[1, 1], [0, 0]],  # P5: A00, A01
        [[-1, 0], [1, 0]], # P6: A10, A00
        [[0, 1], [0, -1]]  # P7: A01, A11
    ]
    
    V_coeffs = [
        [[1, 0], [0, 1]],  # P1: B00, B11
        [[1, 0], [0, 0]],  # P2: B00
        [[0, 1], [0, -1]], # P3: B01, B11
        [[0, 0], [1, -1]], # P4: B10, B00
        [[0, 0], [0, 1]],  # P5: B11
        [[1, 1], [0, 0]],  # P6: B00, B01
        [[0, 0], [1, 1]]   # P7: B10, B11
    ]
    
    W_coeffs = [
        [[1, 0], [0, 1]],  # P1: C00, C11
        [[0, 0], [1, -1]], # P2: C10, C11
        [[0, 1], [0, 1]],  # P3: C01, C11
        [[1, 0], [1, 0]],  # P4: C00, C10
        [[-1, 1], [0, 0]], # P5: C00, C01
        [[0, 0], [0, 1]],  # P6: C11
        [[1, 0], [0, 0]]   # P7: C00
    ]
    
    col_idx = 0
    # Iterate over A-blocks (split along columns of A, which are rows of B)
    # A is 2x4. Split into 2 blocks of 2x2.
    # Block a=0: A cols 0,1. Corresponds to B rows 0,1.
    # Block a=1: A cols 2,3. Corresponds to B rows 2,3.
    for a in range(2):
        # Iterate over B-blocks (split along columns of B)
        # B is 4x6. Split into 3 blocks of 2x2 along columns.
        # Block b=0: B cols 0,1.
        # Block b=1: B cols 2,3.
        # Block b=2: B cols 4,5.
        for b in range(3):
            # Iterate over Strassen terms
            for s in range(7):
                # Update U
                # Local A indices (i, j_loc) -> Global (i, a*2 + j_loc)
                for i in range(2):
                    for j_loc in range(2):
                        coeff = U_coeffs[s][i][j_loc]
                        if coeff != 0:
                            u_idx = i * m + (a * 2 + j_loc)
                            U[u_idx][col_idx] += coeff
                
                # Update V
                # Local B indices (j_loc, k_loc) -> Global (a*2 + j_loc, b*2 + k_loc)
                for j_loc in range(2):
                    for k_loc in range(2):
                        coeff = V_coeffs[s][j_loc][k_loc]
                        if coeff != 0:
                            v_idx = (a * 2 + j_loc) * p + (b * 2 + k_loc)
                            V[v_idx][col_idx] += coeff
                
                # Update W
                # Local C indices (i, k_loc) -> Global (i, b*2 + k_loc)
                # Output format W[k*n + i, r]
                for i in range(2):
                    for k_loc in range(2):
                        coeff = W_coeffs[s][i][k_loc]
                        if coeff != 0:
                            k_global = b * 2 + k_loc
                            w_idx = k_global * n + i
                            W[w_idx][col_idx] += coeff
                
                col_idx += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
