"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (20, rank)
    # W: (p*n, rank) = (10, rank)
    
    rank = 36
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define Strassen 2x2x2 decomposition patterns
    # Each pattern: (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: list of (local_a_row, local_a_col, coeff)
    # V_coeffs: list of (local_b_row, local_b_col, coeff)
    # W_coeffs: list of (local_c_row, local_c_col, coeff)
    
    strassen_patterns = [
        # P1: (A00+A11)(B00+B11)
        ([((0,0,1),(1,1,1)), ((0,0,1),(1,1,1)), ((0,0,1),(1,1,1))]),
        # P2: (A10+A11)B00
        ([((1,0,1),(1,1,1)), ((0,0,1),), ((1,0,1),(1,1,-1))]),
        # P3: A00(B01-B11)
        ([((0,0,1),), ((0,1,1),(1,1,-1)), ((0,1,1),(1,1,1))]),
        # P4: A11(B10-B00)
        ([((1,1,1),), ((1,0,1),(0,0,-1)), ((0,0,1),(1,0,1))]),
        # P5: (A00+A01)B11
        ([((0,0,1),(0,1,1)), ((1,1,1),), ((0,0,-1),(0,1,1))]),
        # P6: (A10-A00)(B00+B01)
        ([((1,0,1),(0,0,-1)), ((0,0,1),(0,1,1)), ((1,1,1),)]),
        # P7: (A01-A11)(B10+B11)
        ([((0,1,1),(1,1,-1)), ((1,0,1),(1,1,1)), ((0,0,1),)]),
    ]
    
    # Define sub-problems
    # Each sub-problem: (a_row_offset, a_col_offset, b_row_offset, b_col_offset, is_2x2x2)
    # a_row_offset: offset for A rows (always 0 since n=2)
    # a_col_offset: offset for A columns
    # b_row_offset: offset for B rows
    # b_col_offset: offset for B columns
    
    subproblems = [
        # (a_col_off, b_row_off, b_col_off, use_strassen)
        (0, 0, 0, True),   # S1: A[:,0:2] x B[0:2,0:2] -> C[:,0:2]
        (0, 0, 2, True),   # S2: A[:,0:2] x B[0:2,2:4] -> C[:,2:4]
        (0, 0, 4, False),  # S3: A[:,0:2] x B[0:2,4:5] -> C[:,4:5]
        (2, 2, 0, True),   # S4: A[:,2:4] x B[2:4,0:2] -> C[:,0:2]
        (2, 2, 2, True),   # S5: A[:,2:4] x B[2:4,2:4] -> C[:,2:4]
        (2, 2, 4, False),  # S6: A[:,2:4] x B[2:4,4:5] -> C[:,4:5]
    ]
    
    for a_col_off, b_row_off, b_col_off, use_strassen in subproblems:
        if use_strassen:
            # 2x2x2 sub-problem, use Strassen (7 terms)
            for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                for (ai, aj, coeff) in u_coeffs:
                    global_u_row = ai * m + (aj + a_col_off)
                    U[global_u_row][col] += coeff
                for (bj, bk, coeff) in v_coeffs:
                    global_v_row = (bj + b_row_off) * p + (bk + b_col_off)
                    V[global_v_row][col] += coeff
                for (ci, ck, coeff) in w_coeffs:
                    global_w_row = (ck + b_col_off) * n + ci
                    W[global_w_row][col] += coeff
                col += 1
        else:
            # 2x2x1 sub-problem, schoolbook (4 terms)
            for ai in range(2):
                for aj in range(2):
                    for bk in range(1):
                        global_u_row = ai * m + (aj + a_col_off)
                        global_v_row = (aj + a_col_off) * p + (b_col_off + bk)  # Wait, this is wrong
                        # B row is b_row_off + aj, B col is b_col_off + bk
                        global_v_row = (b_row_off + aj) * p + (b_col_off + bk)
                        global_w_row = (b_col_off + bk) * n + ai
                        U[global_u_row][col] = 1
                        V[global_v_row][col] = 1
                        W[global_w_row][col] = 1
                        col += 1
    
    # Convert to integers where possible
    def to_int_if_possible(val):
        if abs(val - round(val)) < 1e-9:
            return int(round(val))
        return val
    
    U = [[to_int_if_possible(v) for v in row] for row in U]
    V = [[to_int_if_possible(v) for v in row] for row in V]
    W = [[to_int_if_possible(v) for v in row] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
