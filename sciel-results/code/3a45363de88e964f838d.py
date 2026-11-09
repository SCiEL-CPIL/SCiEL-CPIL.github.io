import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    # u_idx: index in U (i*m + j)
    # v_idx: index in V (j*p + k)
    # w_idx: index in W (k*n + i)
    # coeff_u, coeff_v, coeff_w
    def add_term(u_idx, v_idx, w_idx, cu, cv, cw):
        nonlocal col
        U[u_idx][col] += cu
        V[v_idx][col] += cv
        W[w_idx][col] += cw
        col += 1

    # We have two main blocks: i_block in {0, 1}
    # A_block is rows i_block*2 .. i_block*2+1, all cols
    # B_block is rows i_block*2 .. i_block*2+1, all cols
    
    for i_block in range(2):
        # A indices
        a_row_0 = i_block * 2
        a_row_1 = i_block * 2 + 1
        
        # B indices
        b_row_0 = i_block * 2
        b_row_1 = i_block * 2 + 1
        
        # Split B_block into B_1 (cols 0,1), B_2 (cols 2,3), B_3 (col 4)
        
        # --- Part 1: A_block * B_1 (2x2 * 2x2) using Strassen ---
        # B_1 cols: 0, 1
        # B_1 rows: b_row_0, b_row_1
        
        # Strassen terms for 2x2
        # M1 = A11 * (B00 + B11)
        # A11 is at (a_row_1, 1) -> U idx: a_row_1*4 + 1
        # B00 is at (b_row_0, 0) -> V idx: b_row_0*5 + 0
        # B11 is at (b_row_1, 1) -> V idx: b_row_1*5 + 1
        
        # We need to handle the sums in V.
        # Since V is a matrix, we can just add to the same column.
        
        # M1
        u_idx = a_row_1 * 4 + 1
        v_idx_1 = b_row_0 * 5 + 0
        v_idx_2 = b_row_1 * 5 + 1
        # C indices for M1: C00, C11
        # C00 is at (0, 0) -> W idx: 0*2 + 0 = 0 (relative to block? No, global W)
        # Global W idx: k*n + i.
        # For B_1, k is 0,1.
        # C00: k=0, i=0 -> 0*2+0=0
        # C11: k=1, i=1 -> 1*2+1=3
        # Wait, C is 2x5.
        # C[i, k].
        # W index is k*n + i.
        
        # M1 contributes to C[0,0] and C[1,1] (relative to block start?)
        # No, C is global.
        # A_block is rows 0,1.
        # B_1 is cols 0,1.
        # So C is rows 0,1, cols 0,1.
        # C[0,0] -> W[0*2+0] = 0
        # C[1,1] -> W[1*2+1] = 3
        
        # We need to add to U, V, W for M1.
        # U[u_idx] += 1
        # V[v_idx_1] += 1
        # V[v_idx_2] += 1
        # W[0] += 1
        # W[3] += 1
        
        # But we are building column by column.
        # So we increment col, and set values.
        
        # Let's define a helper to add a Strassen term
        def add_strassen_term(a_row, a_col, b_rows, b_cols, c_offsets, signs):
            # This is getting complex. Let's just hardcode the 7 terms.
            pass

        # Let's do it manually for clarity and speed.
        
        # M1: A11 * (B00 + B11)
        # A11: row a_row_1, col 1
        # B00: row b_row_0, col 0
        # B11: row b_row_1, col 1
        # C00: k=0, i=0
        # C11: k=1, i=1
        
        u = a_row_1 * 4 + 1
        v1 = b_row_0 * 5 + 0
        v2 = b_row_1 * 5 + 1
        w1 = 0 * 2 + 0 # C00
        w2 = 1 * 2 + 1 # C11
        
        U[u][col] = 1
        V[v1][col] = 1
        V[v2][col] = 1
        W[w1][col] = 1
        W[w2][col] = 1
        col += 1

        # M2: (A10 + A11) * B00
        # A10: row a_row_1, col 0
        # A11: row a_row_1, col 1
        # B00: row b_row_0, col 0
        # C10: k=0, i=1
        # C11: k=1, i=1
        
        u1 = a_row_1 * 4 + 0
        u2 = a_row_1 * 4 + 1
        v = b_row_0 * 5 + 0
        w1 = 0 * 2 + 1 # C10
        w2 = 1 * 2 + 1 # C11
        
        U[u1][col] = 1
        U[u2][col] = 1
        V[v][col] = 1
        W[w1][col] = 1
        W[w2][col] = 1
        col += 1

        # M3: A00 * (B10 + B11)
        # A00: row a_row_0, col 0
        # B10: row b_row_1, col 0
        # B11: row b_row_1, col 1
        # C10: k=0, i=1
        # C11: k=1, i=1
        
        u = a_row_0 * 4 + 0
        v1 = b_row_1 * 5 + 0
        v2 = b_row_1 * 5 + 1
        w1 = 0 * 2 + 1 # C10
        w2 = 1 * 2 + 1 # C11
        
        U[u][col] = 1
        V[v1][col] = 1
        V[v2][col] = 1
        W[w1][col] = 1
        W[w2][col] = 1
        col += 1

        # M4: A00 * B00
        # C00, C10
        
        u = a_row_0 * 4 + 0
        v = b_row_0 * 5 + 0
        w1 = 0 * 2 + 0 # C00
        w2 = 0 * 2 + 1 # C10
        
        U[u][col] = 1
        V[v][col] = 1
        W[w1][col] = 1
        W[w2][col] = 1
        col += 1

        # M5: (A00 + A11) * (B11 - B00)
        # A00: row a_row_0, col 0
        # A11: row a_row_1, col 1
        # B11: row b_row_1, col 1
        # B00: row b_row_0, col 0
        # C00: +1, C01: +1, C10: -1, C11: -1
        # C01: k=1, i=0 -> 1*2+0=2
        # C10: k=0, i=1 -> 0*2+1=1
        # C11: k=1, i=1 -> 1*2+1=3
        
        u1 = a_row_0 * 4 + 0
        u2 = a_row_1 * 4 + 1
        v1 = b_row_1 * 5 + 1
        v2 = b_row_0 * 5 + 0
        w00 = 0 * 2 + 0
        w01 = 1 * 2 + 0
        w10 = 0 * 2 + 1
        w11 = 1 * 2 + 1
        
        U[u1][col] = 1
        U[u2][col] = 1
        V[v1][col] = 1
        V[v2][col] = -1
        W[w00][col] = 1
        W[w01][col] = 1
        W[w10][col] = -1
        W[w11][col] = -1
        col += 1

        # M6: (A10 - A00) * (B01 + B11)
        # A10: row a_row_1, col 0
        # A00: row a_row_0, col 0
        # B01: row b_row_0, col 1
        # B11: row b_row_1, col 1
        # C01: +1, C11: +1
        
        u1 = a_row_1 * 4 + 0
        u2 = a_row_0 * 4 + 0
        v1 = b_row_0 * 5 + 1
        v2 = b_row_1 * 5 + 1
        w01 = 1 * 2 + 0
        w11 = 1 * 2 + 1
        
        U[u1][col] = 1
        U[u2][col] = -1
        V[v1][col] = 1
        V[v2][col] = 1
        W[w01][col] = 1
        W[w11][col] = 1
        col += 1

        # M7: (A01 - A11) * (B00 + B10)
        # A01: row a_row_0, col 1
        # A11: row a_row_1, col 1
        # B00: row b_row_0, col 0
        # B10: row b_row_1, col 0
        # C00: +1, C10: +1
        
        u1 = a_row_0 * 4 + 1
        u2 = a_row_1 * 4 + 1
        v1 = b_row_0 * 5 + 0
        v2 = b_row_1 * 5 + 0
        w00 = 0 * 2 + 0
        w10 = 0 * 2 + 1
        
        U[u1][col] = 1
        U[u2][col] = -1
        V[v1][col] = 1
        V[v2][col] = 1
        W[w00][col] = 1
        W[w10][col] = 1
        col += 1

        # --- Part 2: A_block * B_2 (2x2 * 2x2) using Strassen ---
        # B_2 cols: 2, 3
        # Same logic, but B cols are 2,3.
        
        # M1
        u = a_row_1 * 4 + 1
        v1 = b_row_0 * 5 + 2
        v2 = b_row_1 * 5 + 3
        w1 = 2 * 2 + 0 # C02
        w2 = 3 * 2 + 1 # C13
        
        U[u][col] = 1
        V[v1][col] = 1
        V[v2][col] = 1
        W[w1][col] = 1
        W[w2][col] = 1
        col += 1

        # M2
        u1 = a_row_1 * 4 + 0
        u2 = a_row_1 * 4 + 1
        v = b_row_0 * 5 + 2
        w1 = 2 * 2 + 1 # C12
        w2 = 3 * 2 + 1 # C13
        
        U[u1][col] = 1
        U[u2][col] = 1
        V[v][col] = 1
        W[w1][col] = 1
        W[w2][col] = 1
        col += 1

        # M3
        u = a_row_0 * 4 + 0
        v1 = b_row_1 * 5 + 2
        v2 = b_row_1 * 5 + 3
        w1 = 2 * 2 + 1 # C12
        w2 = 3 * 2 + 1 # C13
        
        U[u][col] = 1
        V[v1][col] = 1
        V[v2][col] = 1
        W[w1][col] = 1
        W[w2][col] = 1
        col += 1

        # M4
        u = a_row_0 * 4 + 0
        v = b_row_0 * 5 + 2
        w1 = 2 * 2 + 0 # C02
        w2 = 2 * 2 + 1 # C12
        
        U[u][col] = 1
        V[v][col] = 1
        W[w1][col] = 1
        W[w2][col] = 1
        col += 1

        # M5
        u1 = a_row_0 * 4 + 0
        u2 = a_row_1 * 4 + 1
        v1 = b_row_1 * 5 + 3
        v2 = b_row_0 * 5 + 2
        w00 = 2 * 2 + 0 # C02
        w01 = 3 * 2 + 0 # C03
        w10 = 2 * 2 + 1 # C12
        w11 = 3 * 2 + 1 # C13
        
        U[u1][col] = 1
        U[u2][col] = 1
        V[v1][col] = 1
        V[v2][col] = -1
        W[w00][col] = 1
        W[w01][col] = 1
        W[w10][col] = -1
        W[w11][col] = -1
        col += 1

        # M6
        u1 = a_row_1 * 4 + 0
        u2 = a_row_0 * 4 + 0
        v1 = b_row_0 * 5 + 3
        v2 = b_row_1 * 5 + 3
        w01 = 3 * 2 + 0 # C03
        w11 = 3 * 2 + 1 # C13
        
        U[u1][col] = 1
        U[u2][col] = -1
        V[v1][col] = 1
        V[v2][col] = 1
        W[w01][col] = 1
        W[w11][col] = 1
        col += 1

        # M7
        u1 = a_row_0 * 4 + 1
        u2 = a_row_1 * 4 + 1
        v1 = b_row_0 * 5 + 2
        v2 = b_row_1 * 5 + 2
        w00 = 2 * 2 + 0 # C02
        w10 = 2 * 2 + 1 # C12
        
        U[u1][col] = 1
        U[u2][col] = -1
        V[v1][col] = 1
        V[v2][col] = 1
        W[w00][col] = 1
        W[w10][col] = 1
        col += 1

        # --- Part 3: A_block * B_3 (2x2 * 2x1) Schoolbook ---
        # B_3 col: 4
        # C col: 4
        
        # A00 * B04
        u = a_row_0 * 4 + 0
        v = b_row_0 * 5 + 4
        w = 4 * 2 + 0 # C04
        U[u][col] = 1
        V[v][col] = 1
        W[w][col] = 1
        col += 1

        # A01 * B14
        u = a_row_0 * 4 + 1
        v = b_row_1 * 5 + 4
        w = 4 * 2 + 0 # C04
        U[u][col] = 1
        V[v][col] = 1
        W[w][col] = 1
        col += 1

        # A10 * B04
        u = a_row_1 * 4 + 0
        v = b_row_0 * 5 + 4
        w = 4 * 2 + 1 # C14
        U[u][col] = 1
        V[v][col] = 1
        W[w][col] = 1
        col += 1

        # A11 * B14
        u = a_row_1 * 4 + 1
        v = b_row_1 * 5 + 4
        w = 4 * 2 + 1 # C14
        U[u][col] = 1
        V[v][col] = 1
        W[w][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
