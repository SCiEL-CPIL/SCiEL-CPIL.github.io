import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=4
    # Rank 44 decomposition using Strassen for top 2 rows and schoolbook for bottom 1 row.
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    # U index: i * m + j
    # V index: j * p + k
    # W index: k * n + i
    def add_term(i, j, k, val=1):
        nonlocal col
        U[i * m + j][col] += val
        V[j * p + k][col] += val
        W[k * n + i][col] += val
        col += 1

    # --- Top 2 Rows (i=0, 1) ---
    # A_top is 2x4. B is 4x4.
    # Split B into 2x2 blocks:
    # B11 (rows 0-1, cols 0-1), B12 (rows 0-1, cols 2-3)
    # B21 (rows 2-3, cols 0-1), B22 (rows 2-3, cols 2-3)
    # Split A_top into A_L (cols 0-1) and A_R (cols 2-3).
    # A_L is 2x2, A_R is 2x2.
    
    # We need to compute:
    # C_TL = A_L B11 + A_R B21
    # C_TR = A_L B12 + A_R B22
    
    # We will use 7 Strassen products for A_L B11, 7 for A_R B21, 7 for A_L B12, 7 for A_R B22.
    # Total 28 terms.
    
    # Strassen coefficients for X*Y:
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)y00
    # M3 = x00(y01-y11)
    # M4 = x11(y10-y00)
    # M5 = (x00+x01)y11
    # M6 = (x10-x00)(y00+y01)
    # M7 = (x01-x11)(y10+y11)
    # P00 = M1 + M4 - M5 + M7
    # P01 = M3 + M5
    # P10 = M2 + M4
    # P11 = M1 - M2 + M3 + M6
    
    # We define a function to add the 7 terms for X*Y where X is at A(i, j) and Y is at B(j, k)
    # But here X and Y are 2x2 blocks.
    # Let's manually map the scalar indices.
    
    # Block A_L: i in {0,1}, j in {0,1}
    # Block A_R: i in {0,1}, j in {2,3}
    # Block B11: j in {0,1}, k in {0,1}
    # Block B12: j in {0,1}, k in {2,3}
    # Block B21: j in {2,3}, k in {0,1}
    # Block B22: j in {2,3}, k in {2,3}
    
    # For each 2x2 matrix product, we add 7 terms.
    # The "scalar" indices within the 2x2 block are 0,1.
    
    def add_strassen_block(i_base, j_base, k_base, sign=1):
        # Adds 7 terms for the product of A[i_base:i_base+2, j_base:j_base+2] * B[j_base:j_base+2, k_base:k_base+2]
        # i_base, j_base, k_base are the top-left indices of the 2x2 blocks.
        # The result contributes to C[i_base:i_base+2, k_base:k_base+2].
        # We need to add the contributions of M1..M7 to the correct (i,k) positions in W.
        # And set U and V accordingly.
        
        # M1
        # U: (i_base, j_base) + (i_base+1, j_base+1)
        # V: (j_base, k_base) + (j_base+1, k_base+1)
        # W: (k_base, i_base) + (k_base, i_base+1) + (k_base+1, i_base) + (k_base+1, i_base+1) ? 
        # No, W index is k*n + i.
        # The term is (x00+x11)(y00+y11).
        # x00 is A[i0, j0], x11 is A[i1, j1].
        # y00 is B[j0, k0], y11 is B[j1, k1].
        # Product is scalar.
        # It contributes to P00, P11.
        # P00 is at C[i0, k0].
        # P11 is at C[i1, k1].
        # So W[k0, i0] += 1, W[k1, i1] += 1.
        
        # This manual expansion is error-prone. 
        # Instead, we can generate the U, V, W entries by iterating over the 7 formulas and the 4 output positions.
        
        # Let's define the 7 linear combinations for U and V.
        # For a block product X*Y -> P:
        # X is 2x2, Y is 2x2.
        # U entries are from X, V entries are from Y.
        
        # M1: U = x00 + x11, V = y00 + y11. P00 += 1, P11 += 1.
        # M2: U = x10 + x11, V = y00. P10 += 1.
        # M3: U = x00, V = y01 - y11. P01 += 1.
        # M4: U = x11, V = y10 - y00. P00 += 1, P10 += 1.
        # M5: U = x00 + x01, V = y11. P00 -= 1, P01 += 1.
        # M6: U = x10 - x00, V = y00 + y01. P11 += 1.
        # M7: U = x01 - x11, V = y10 + y11. P00 += 1, P11 -= 1.
        
        # We will iterate through these 7 terms.
        
        terms = [
            # (u_coeffs, v_coeffs, w_coeffs)
            # u_coeffs: dict {(i,j): coeff}
            # v_coeffs: dict {(j,k): coeff}
            # w_coeffs: dict {(k,i): coeff}
            
            # M1
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            # M2
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
            # M3
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}),
            # M4
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
            # M5
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            # M6
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            # M7
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):-1}),
        ]
        
        for u_c, v_c, w_c in terms:
            # Scale by sign
            for (di, dj), cu in u_c.items():
                U[(i_base + di) * m + (j_base + dj)][col] += sign * cu
            for (dj, dk), cv in v_c.items():
                V[(j_base + dj) * p + (k_base + dk)][col] += sign * cv
            for (dk, di), cw in w_c.items():
                W[(k_base + dk) * n + (i_base + di)][col] += sign * cw
            col += 1

    # Apply Strassen for A_L * B11 (i=0,1; j=0,1; k=0,1)
    add_strassen_block(0, 0, 0, 1)
    # Apply Strassen for A_R * B21 (i=0,1; j=2,3; k=0,1)
    add_strassen_block(0, 2, 0, 1)
    # Apply Strassen for A_L * B12 (i=0,1; j=0,1; k=2,3)
    add_strassen_block(0, 0, 2, 1)
    # Apply Strassen for A_R * B22 (i=0,1; j=2,3; k=2,3)
    add_strassen_block(0, 2, 2, 1)
    
    # --- Bottom Row (i=2) ---
    # A_bot is 1x4.
    # C_BL = A_L2 B11 + A_R2 B21
    # A_L2 is 1x2 (i=2, j=0,1). B11 is 2x2 (j=0,1; k=0,1).
    # Product is 1x2.
    # C_BL,0 = A[2,0]B[0,0] + A[2,1]B[1,0] + A[2,2]B[2,0] + A[2,3]B[3,0]
    # This is 4 scalar mults.
    # C_BL,1 = A[2,0]B[0,1] + A[2,1]B[1,1] + A[2,2]B[2,1] + A[2,3]B[3,1]
    # This is 4 scalar mults.
    # Total 8 mults.
    
    # We just add 8 rank-1 terms.
    # Term 1: i=2, j=0, k=0
    add_term(2, 0, 0, 1)
    # Term 2: i=2, j=1, k=0
    add_term(2, 1, 0, 1)
    # Term 3: i=2, j=2, k=0
    add_term(2, 2, 0, 1)
    # Term 4: i=2, j=3, k=0
    add_term(2, 3, 0, 1)
    
    # Term 5: i=2, j=0, k=1
    add_term(2, 0, 1, 1)
    # Term 6: i=2, j=1, k=1
    add_term(2, 1, 1, 1)
    # Term 7: i=2, j=2, k=1
    add_term(2, 2, 1, 1)
    # Term 8: i=2, j=3, k=1
    add_term(2, 3, 1, 1)
    
    # Term 9: i=2, j=0, k=2
    add_term(2, 0, 2, 1)
    # Term 10: i=2, j=1, k=2
    add_term(2, 1, 2, 1)
    # Term 11: i=2, j=2, k=2
    add_term(2, 2, 2, 1)
    # Term 12: i=2, j=3, k=2
    add_term(2, 3, 2, 1)
    
    # Term 13: i=2, j=0, k=3
    add_term(2, 0, 3, 1)
    # Term 14: i=2, j=1, k=3
    add_term(2, 1, 3, 1)
    # Term 15: i=2, j=2, k=3
    add_term(2, 2, 3, 1)
    # Term 16: i=2, j=3, k=3
    add_term(2, 3, 3, 1)
    
    # Total terms: 28 + 16 = 44.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
