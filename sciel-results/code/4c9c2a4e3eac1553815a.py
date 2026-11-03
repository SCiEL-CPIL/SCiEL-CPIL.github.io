"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=6
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 rows
    V = [[0] * rank for _ in range(m * p)]  # 24 rows
    W = [[0] * rank for _ in range(p * n)]  # 12 rows
    
    col = 0
    
    def add_term(a_coeffs, b_coeffs, c_coeffs):
        nonlocal col
        for (ai, aj), ac in a_coeffs.items():
            U[ai * m + aj][col] += ac
        for (bj, bk), bc in b_coeffs.items():
            V[bj * p + bk][col] += bc
        for (ci, ck), cc in c_coeffs.items():
            W[ck * n + ci][col] += cc
        col += 1
    
    # 6 sub-problems: (a_col_start, b_row_start, b_col_start, c_col_start)
    sub_problems = [
        (0, 0, 0, 0),
        (2, 2, 0, 0),
        (0, 0, 2, 2),
        (2, 2, 2, 2),
        (0, 0, 4, 4),
        (2, 2, 4, 4),
    ]
    
    for a_col, b_row, b_col, c_col in sub_problems:
        # M1 = (A00 + A11)(B00 + B11)
        add_term(
            a_coeffs={(0, a_col): 1, (1, a_col + 1): 1},
            b_coeffs={(b_row, b_col): 1, (b_row + 1, b_col + 1): 1},
            c_coeffs={(0, c_col): 1, (1, c_col + 1): 1}
        )
        
        # M2 = (A10 + A11)B00
        add_term(
            a_coeffs={(1, a_col): 1, (1, a_col + 1): 1},
            b_coeffs={(b_row, b_col): 1},
            c_coeffs={(1, c_col): 1}
        )
        
        # M3 = A00(B01 - B11)
        add_term(
            a_coeffs={(0, a_col): 1},
            b_coeffs={(b_row, b_col + 1): 1, (b_row + 1, b_col + 1): -1},
            c_coeffs={(0, c_col + 1): 1, (1, c_col + 1): 1}
        )
        
        # M4 = A11(B10 - B00)
        add_term(
            a_coeffs={(1, a_col + 1): 1},
            b_coeffs={(b_row + 1, b_col): 1, (b_row, b_col): -1},
            c_coeffs={(0, c_col): 1, (1, c_col): 1}
        )
        
        # M5 = (A00 + A01)B11
        add_term(
            a_coeffs={(0, a_col): 1, (0, a_col + 1): 1},
            b_coeffs={(b_row + 1, b_col + 1): 1},
            c_coeffs={(0, c_col + 1): 1, (0, c_col): -1}
        )
        
        # M6 = (A10 - A00)(B00 + B01)
        add_term(
            a_coeffs={(1, a_col): 1, (0, a_col): -1},
            b_coeffs={(b_row, b_col): 1, (b_row, b_col + 1): 1},
            c_coeffs={(1, c_col + 1): 1}
        )
        
        # M7 = (A01 - A11)(B10 + B11)
        add_term(
            a_coeffs={(0, a_col + 1): 1, (1, a_col + 1): -1},
            b_coeffs={(b_row + 1, b_col): 1, (b_row + 1, b_col + 1): 1},
            c_coeffs={(0, c_col): 1}
        )
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
