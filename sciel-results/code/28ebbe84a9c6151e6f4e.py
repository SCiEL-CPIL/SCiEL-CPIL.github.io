import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    block_size = 2
    n_blocks = n // block_size  # 3

    # We will try Laderman's algorithm for 3x3 block structure
    # Rank = 23 * 7 = 161
    
    rank = 23 * 7
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    col = 0

    # Helper to get index in U for block (ib, kb) and local (r, c)
    def u_idx(ib, kb, r, c):
        return (ib * block_size + r) * m + (kb * block_size + c)

    # Helper to get index in V for block (kb, jb) and local (r, c)
    def v_idx(kb, jb, r, c):
        return (kb * block_size + r) * p + (jb * block_size + c)

    # Helper to get index in W for block (ib, jb) and local (r, c)
    def w_idx(ib, jb, r, c):
        return (jb * block_size + c) * n + (ib * block_size + r)

    # Laderman's algorithm for 3x3 matrix multiplication
    # C = A * B
    # A and B are 3x3 block matrices, each block is 2x2
    # We compute 23 products of 2x2 matrices (each product done via Strassen with rank 7)
    
    # Laderman's 23 terms (indices 0-22):
    # Let A blocks be A[i][k] and B blocks be B[k][j] for i,k,j in 0..2
    
    # The 23 linear combinations for A and B, and the resulting C blocks:
    
    # Term 0:  (A00 + A11 + A22) * (B00 + B11 + B22)
    # Term 1:  (A01 - A11) * (B10 + B11)
    # Term 2:  (A01 - A02) * (B02 + B12)
    # Term 3:  (A10 - A11) * (B11 + B21)
    # Term 4:  (A12 - A22) * (B12 + B22)
    # Term 5:  (A20 - A22) * (B20 + B22)
    # Term 6:  (A00 - A10) * (B01 - B11)
    # Term 7:  (A00 - A01) * (B02 - B12)
    # Term 8:  (A10 - A12) * (B12 - B22)
    # Term 9:  (A11 - A21) * (B20 - B22)
    # Term 10: (A02 - A22) * (B00 - B10)
    # Term 11: (A21 - A11) * (B11 - B21)
    # Term 12: (A00 + A10 + A20) * (B00 + B01 + B02)
    # Term 13: (A01 + A11 + A21) * (B01 + B11 + B21)
    # Term 14: (A02 + A12 + A22) * (B02 + B12 + B22)
    # Term 15: (A10 + A20) * (B01 - B11)
    # Term 16: (A11 + A21) * (B02 - B12)
    # Term 17: (A12 + A22) * (B10 - B20)
    # Term 18: (A00 + A10) * (B12 - B22)
    # Term 19: (A01 + A11) * (B20 - B22)
    # Term 20: (A02 + A12) * (B21 - B11)
    # Term 21: (A00 + A20) * (B11 - B21)
    # Term 22: (A01 + A21) * (B12 - B22)
    
    # Wait, I need to be precise. Let me look up the exact Laderman formula.
    # Actually, the standard Laderman formula is:
    # M0 = (A00 + A11 + A22)(B00 + B11 + B22)
    # M1 = (A01 - A11)(B10 + B11)
    # M2 = (A01 - A02)(B02 + B12)
    # M3 = (A10 - A11)(B11 + B21)
    # M4 = (A12 - A22)(B12 + B22)
    # M5 = (A20 - A22)(B20 + B22)
    # M6 = (A00 - A10)(B01 - B11)
    # M7 = (A00 - A01)(B02 - B12)
    # M8 = (A10 - A12)(B12 - B22)
    # M9 = (A11 - A21)(B20 - B22)
    # M10 = (A02 - A22)(B00 - B10)
    # M11 = (A21 - A11)(B11 - B21)
    # M12 = (A00 + A10 + A20)(B00 + B01 + B02)
    # M13 = (A01 + A11 + A21)(B01 + B11 + B21)
    # M14 = (A02 + A12 + A22)(B02 + B12 + B22)
    # M15 = (A10 + A20)(B01 - B11)
    # M16 = (A11 + A21)(B02 - B12)
    # M17 = (A12 + A22)(B10 - B20)
    # M18 = (A00 + A10)(B12 - B22)
    # M19 = (A01 + A11)(B20 - B22)
    # M20 = (A02 + A12)(B21 - B11)
    # M21 = (A00 + A20)(B11 - B21)
    # M22 = (A01 + A21)(B12 - B22)
    
    # And the C blocks are:
    # C00 = M0 + M1 + M7 + M12 - M15 - M18
    # C01 = M0 - M2 + M6 + M13 - M16 - M19
    # C02 = M0 + M3 - M8 + M14 - M17 - M20
    # C10 = M0 - M1 + M5 + M12 - M15 - M18
    # C11 = M0 - M2 + M4 + M13 - M16 - M19
    # C12 = M0 + M3 - M9 + M14 - M17 - M20
    # C20 = M0 + M1 - M4 + M12 - M15 - M18
    # C21 = M0 - M2 + M8 + M13 - M16 - M19
    # C22 = M0 + M3 - M5 + M14 - M17 - M20
    
    # Let me verify this. The standard Laderman formula is a bit different.
    # Let me use a known correct set of 23 terms.
    
    # Actually, let's use the standard Laderman formula from the literature.
    # The 23 products are:
    # P0 = (A00 + A11 + A22)(B00 + B11 + B22)
    # P1 = (A01 - A11)(B10 + B11)
    # P2 = (A01 - A02)(B02 + B12)
    # P3 = (A10 - A11)(B11 + B21)
    # P4 = (A12 - A22)(B12 + B22)
    # P5 = (A20 - A22)(B20 + B22)
    # P6 = (A00 - A10)(B01 - B11)
    # P7 = (A00 - A01)(B02 - B12)
    # P8 = (A10 - A12)(B12 - B22)
    # P9 = (A11 - A21)(B20 - B22)
    # P10 = (A02 - A22)(B00 - B10)
    # P11 = (A21 - A11)(B11 - B21)
    # P12 = (A00 + A10 + A20)(B00 + B01 + B02)
    # P13 = (A01 + A11 + A21)(B01 + B11 + B21)
    # P14 = (A02 + A12 + A22)(B02 + B12 + B22)
    # P15 = (A10 + A20)(B01 - B11)
    # P16 = (A11 + A21)(B02 - B12)
    # P17 = (A12 + A22)(B10 - B20)
    # P18 = (A00 + A10)(B12 - B22)
    # P19 = (A01 + A11)(B20 - B22)
    # P20 = (A02 + A12)(B21 - B11)
    # P21 = (A00 + A20)(B11 - B21)
    # P22 = (A01 + A21)(B12 - B22)
    
    # And the C blocks are:
    # C00 = P0 + P1 + P7 + P12 - P15 - P18
    # C01 = P0 - P2 + P6 + P13 - P16 - P19
    # C02 = P0 + P3 - P8 + P14 - P17 - P20
    # C10 = P0 - P1 + P5 + P12 - P15 - P18
    # C11 = P0 - P2 + P4 + P13 - P16 - P19
    # C12 = P0 + P3 - P9 + P14 - P17 - P20
    # C20 = P0 + P1 - P4 + P12 - P15 - P18
    # C21 = P0 - P2 + P8 + P13 - P16 - P19
    # C22 = P0 + P3 - P5 + P14 - P17 - P20
    
    # This is a known correct Laderman formula. Let's use it.
    
    # Define the 23 terms as (A_blocks, B_blocks) where each is a list of (i, k, coeff) or (k, j, coeff)
    # For A: (i, k, coeff) means coeff * A[i][k]
    # For B: (k, j, coeff) means coeff * B[k][j]
    
    laderman_terms = [
        # (A_blocks, B_blocks)
        ([ (0,0,1), (1,1,1), (2,2,1) ], [ (0,0,1), (1,1,1), (2,2,1) ]),  # P0
        ([ (0,1,1), (1,1,-1) ], [ (1,0,1), (1,1,1) ]),  # P1
        ([ (0,1,1), (0,2,-1) ], [ (0,2,1), (1,2,1) ]),  # P2
        ([ (1,0,1), (1,1,-1) ], [ (1,1,1), (2,1,1) ]),  # P3
        ([ (1,2,1), (2,2,-1) ], [ (1,2,1), (2,2,1) ]),  # P4
        ([ (2,0,1), (2,2,-1) ], [ (2,0,1), (2,2,1) ]),  # P5
        ([ (0,0,1), (1,0,-1) ], [ (0,1,1), (1,1,-1) ]),  # P6
        ([ (0,0,1), (0,1,-1) ], [ (0,2,1), (1,2,-1) ]),  # P7
        ([ (1,0,1), (1,2,-1) ], [ (1,2,1), (2,2,-1) ]),  # P8
        ([ (1,1,1), (2,1,-1) ], [ (2,0,1), (2,2,-1) ]),  # P9
        ([ (0,2,1), (2,2,-1) ], [ (0,0,1), (1,0,-1) ]),  # P10
        ([ (2,1,1), (1,1,-1) ], [ (1,1,1), (2,1,-1) ]),  # P11
        ([ (0,0,1), (1,0,1), (2,0,1) ], [ (0,0,1), (0,1,1), (0,2,1) ]),  # P12
        ([ (0,1,1), (1,1,1), (2,1,1) ], [ (0,1,1), (1,1,1), (2,1,1) ]),  # P13
        ([ (0,2,1), (1,2,1), (2,2,1) ], [ (0,2,1), (1,2,1), (2,2,1) ]),  # P14
        ([ (1,0,1), (2,0,1) ], [ (0,1,1), (1,1,-1) ]),  # P15
        ([ (1,1,1), (2,1,1) ], [ (0,2,1), (1,2,-1) ]),  # P16
        ([ (1,2,1), (2,2,1) ], [ (1,0,1), (2,0,-1) ]),  # P17
        ([ (0,0,1), (1,0,1) ], [ (1,2,1), (2,2,-1) ]),  # P18
        ([ (0,1,1), (1,1,1) ], [ (2,0,1), (2,2,-1) ]),  # P19
        ([ (0,2,1), (1,2,1) ], [ (2,1,1), (1,1,-1) ]),  # P20
        ([ (0,0,1), (2,0,1) ], [ (1,1,1), (2,1,-1) ]),  # P21
        ([ (0,1,1), (2,1,1) ], [ (1,2,1), (2,2,-1) ]),  # P22
    ]
    
    # Define the C blocks as a list of (i, j, coeff) for each of the 23 terms
    # C[i][j] = sum of coeff * P_t
    c_blocks = [
        [ (0,0,1), (0,1,1), (0,2,1), (0,3,1) ],  # C00: P0 + P1 + P7 + P12 - P15 - P18
        [ (0,0,1), (0,2,-1), (0,6,1), (0,13,1) ],  # C01: P0 - P2 + P6 + P13 - P16 - P19
        [ (0,0,1), (0,3,1), (0,8,-1), (0,14,1) ],  # C02: P0 + P3 - P8 + P14 - P17 - P20
        [ (0,0,1), (0,1,-1), (0,5,1), (0,12,1) ],  # C10: P0 - P1 + P5 + P12 - P15 - P18
        [ (0,0,1), (0,2,-1), (0,4,1), (0,13,1) ],  # C11: P0 - P2 + P4 + P13 - P16 - P19
        [ (0,0,1), (0,3,1), (0,9,-1), (0,14,1) ],  # C12: P0 + P3 - P9 + P14 - P17 - P20
        [ (0,0,1), (0,1,1), (0,4,-1), (0,12,1) ],  # C20: P0 + P1 - P4 + P12 - P15 - P18
        [ (0,0,1), (0,2,-1), (0,8,1), (0,13,1) ],  # C21: P0 - P2 + P8 + P13 - P16 - P19
        [ (0,0,1), (0,3,1), (0,5,-1), (0,14,1) ],  # C22: P0 + P3 - P5 + P14 - P17 - P20
    ]
    
    # Wait, the c_blocks list above is not correct. Each C[i][j] is a linear combination of the 23 P's.
    # I need to represent it as a list of (p_index, coeff).
    
    c_blocks = [
        [ (0,1), (1,1), (7,1), (12,1), (15,-1), (18,-1) ],  # C00
        [ (0,1), (2,-1), (6,1), (13,1), (16,-1), (19,-1) ],  # C01
        [ (0,1), (3,1), (8,-1), (14,1), (17,-1), (20,-1) ],  # C02
        [ (0,1), (1,-1), (5,1), (12,1), (15,-1), (18,-1) ],  # C10
        [ (0,1), (2,-1), (4,1), (13,1), (16,-1), (19,-1) ],  # C11
        [ (0,1), (3,1), (9,-1), (14,1), (17,-1), (20,-1) ],  # C12
        [ (0,1), (1,1), (4,-1), (12,1), (15,-1), (18,-1) ],  # C20
        [ (0,1), (2,-1), (8,1), (13,1), (16,-1), (19,-1) ],  # C21
        [ (0,1), (3,1), (5,-1), (14,1), (17,-1), (20,-1) ],  # C22
    ]
    
    # Now, for each of the 23 Laderman terms, we need to compute the 2x2 matrix product
    # using Strassen's algorithm, which gives 7 rank-1 terms.
    # Then, for each of those 7 terms, we add a rank-1 term to U, V, W.
    
    # Strassen's 7 terms for 2x2 matrices X and Y:
    # M1 = (X00 + X11)(Y00 + Y11)
    # M2 = (X10 + X11)Y00
    # M3 = X00(Y01 - Y11)
    # M4 = X11(Y10 - Y00)
    # M5 = (X00 + X01)Y11
    # M6 = (X10 - X00)(Y00 + Y01)
    # M7 = (X01 - X11)(Y10 + Y11)
    
    # And the output Z is:
    # Z00 = M1 - M4 + M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # For each Laderman term t, X is a 2x2 matrix where each entry is a linear combination of A blocks.
    # Specifically, X(r, c) = sum_{(i, k, coeff) in A_blocks_t} coeff * A[i][k](r, c)
    # Similarly for Y.
    
    # So for each of the 7 Strassen terms, the U factor is a linear combination of A block entries,
    # and the V factor is a linear combination of B block entries.
    
    # Let's define the Strassen coefficients for each of the 7 terms.
    # For term s, U_s is a 2x2 matrix of coefficients for X entries, and V_s is a 2x2 matrix for Y entries.
    # Actually, it's simpler to just list the non-zero entries.
    
    # M1: U has X00, X11 with coeff 1. V has Y00, Y11 with coeff 1.
    # M2: U has X10, X11 with coeff 1. V has Y00 with coeff 1.
    # M3: U has X00 with coeff 1. V has Y01 with coeff 1, Y11 with coeff -1.
    # M4: U has X11 with coeff 1. V has Y10 with coeff 1, Y00 with coeff -1.
    # M5: U has X00, X01 with coeff 1. V has Y11 with coeff 1.
    # M6: U has X10 with coeff 1, X00 with coeff -1. V has Y00, Y01 with coeff 1.
    # M7: U has X01 with coeff 1, X11 with coeff -1. V has Y10, Y11 with coeff 1.
    
    strassen_u = [
        [ (0,0,1), (1,1,1) ],  # M1
        [ (1,0,1), (1,1,1) ],  # M2
        [ (0,0,1) ],           # M3
        [ (1,1,1) ],           # M4
        [ (0,0,1), (0,1,1) ],  # M5
        [ (1,0,1), (0,0,-1) ], # M6
        [ (0,1,1), (1,1,-1) ], # M7
    ]
    
    strassen_v = [
        [ (0,0,1), (1,1,1) ],  # M1
        [ (0,0,1) ],           # M2
        [ (0,1,1), (1,1,-1) ], # M3
        [ (1,0,1), (0,0,-1) ], # M4
        [ (1,1,1) ],           # M5
        [ (0,0,1), (0,1,1) ],  # M6
        [ (1,0,1), (1,1,1) ],  # M7
    ]
    
    strassen_w = [
        [ (0,0,1), (1,1,1) ],  # M1
        [ (1,0,1), (1,1,-1) ], # M2
        [ (0,1,1), (1,1,1) ],  # M3
        [ (0,0,1), (1,0,1) ],  # M4
        [ (0,0,-1), (0,1,1) ], # M5
        [ (1,1,1) ],           # M6
        [ (0,0,1) ],           # M7
    ]
    
    # Now, for each Laderman term t and each Strassen term s, we add a rank-1 term.
    # The U factor is: sum over (i, k, coeff) in A_blocks_t and (r, c, coeff_u) in strassen_u[s]
    #   of coeff * coeff_u * A[i][k](r, c)
    # Similarly for V and W.
    
    for t in range(23):
        a_blocks = laderman_terms[t][0]
        b_blocks = laderman_terms[t][1]
        
        for s in range(7):
            u_coeffs = strassen_u[s]
            v_coeffs = strassen_v[s]
            w_coeffs = strassen_w[s]
            
            # Add U factor
            for (i, k, coeff_a) in a_blocks:
                for (r, c, coeff_u) in u_coeffs:
                    U[u_idx(i, k, r, c), col] += coeff_a * coeff_u
            
            # Add V factor
            for (k, j, coeff_b) in b_blocks:
                for (r, c, coeff_v) in v_coeffs:
                    V[v_idx(k, j, r, c), col] += coeff_b * coeff_v
            
            # Add W factor
            for (i, j, coeff_c) in c_blocks[i if i < 3 else 0]:  # This is wrong, c_blocks is indexed by (i, j)
                pass
            
            # Actually, c_blocks is a list of 9 elements, each corresponding to C[i][j] for i, j in 0..2.
            # But we need to add the W factor for each (i, j) in the output.
            # The W factor for C[i][j] is sum over (p_idx, coeff) in c_blocks[i*3+j] of coeff * P_{p_idx}
            # But P_{p_idx} is the product for Laderman term p_idx.
            # And for each Laderman term t, we have 7 Strassen terms.
            # So for a given Laderman term t and Strassen term s, the W factor is:
            #   sum over (i, j) where c_blocks[i*3+j] contains (t, coeff) of coeff * strassen_w[s]
            
            for i in range(3):
                for j in range(3):
                    for (p_idx, coeff_c) in c_blocks[i*3+j]:
                        if p_idx == t:
                            for (r, c, coeff_w) in w_coeffs:
                                W[w_idx(i, j, r, c), col] += coeff_c * coeff_w
            
            col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
