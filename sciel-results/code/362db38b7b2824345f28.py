"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 130.0
    n, m, p = DIMENSIONS  # 2, 4, 8

    # Strassen 2x2x2: rank 7
    # For 2x2 A, 2x2 B -> 2x2 C:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    # C00 = P1+P4-P5+P7
    # C01 = P3+P5
    # C10 = P2+P4
    # C11 = P1-P2+P3+P6

    # Each Strassen product r has:
    #   U_coeffs: dict {(i,j): coeff} for A indices
    #   V_coeffs: dict {(j,k): coeff} for B indices  
    #   W_coeffs: dict {(k,i): coeff} for C indices
    strassen_products = [
        # P1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # P2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}),
        # P3
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),
        # P4
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
        # P5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}),
        # P6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # P7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    # For 2x4 * 4x8 -> 2x8:
    # Split into 8 independent 2x2 * 2x2 blocks:
    # Block b: A_cols = 2*(b%2)..2*(b%2)+1, B_rows = same, B_cols = 2*(b//2)..2*(b//2)+1
    # 8 blocks * 7 Strassen = 56 rank
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56

    # Helper to add a Strassen block
    def add_strassen_block(a_col0, b_row0, b_col0, col_start):
        c = col_start
        for u_coeff, v_coeff, w_coeff in strassen_products:
            for (li, lj), coeff in u_coeff.items():
                gi = li
                gj = a_col0 + lj
                U[gi * m + gj][c] = coeff
            for (lj, lk), coeff in v_coeff.items():
                gj = b_row0 + lj
                gk = b_col0 + lk
                V[gj * p + gk][c] = coeff
            for (lk, li), coeff in w_coeff.items():
                gk = b_col0 + lk
                gi = li
                W[gk * n + gi][c] = coeff
            c += 1
        return c

    col = 0
    # 8 blocks: (a_col0, b_row0, b_col0)
    blocks = [
        (0, 0, 0),
        (2, 2, 0),
        (0, 0, 2),
        (2, 2, 2),
        (0, 0, 4),
        (2, 2, 4),
        (0, 0, 6),
        (2, 2, 6),
    ]
    
    for a_col0, b_row0, b_col0 in blocks:
        col = add_strassen_block(a_col0, b_row0, b_col0, col)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
