"""Program for matrix multiplication 6x6x6 tensor decomposition.
Uses 2-level recursive Strassen: 6x6 -> 7 * 3x3, each 3x3 -> 7 * (2x1 or 1x1) products.
Total rank = 7 * 7 * 4 = 196? No, let's use the clean 7*27=189 approach.
Actually, let me implement the rank-189 decomposition cleanly.
"""
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-189 exact tensor decomposition for 6x6 matrix multiplication.
    
    Approach: 1-level Strassen on 6x6 split into 2x2 blocks of 3x3.
    Each of the 7 Strassen terms is a 3x3 matrix multiplication, decomposed
    via schoolbook (rank 27). Total rank = 7 * 27 = 189.
    
    U: shape (36, 189) - factors for input A
    V: shape (36, 189) - factors for input B  
    W: shape (36, 189) - factors for output C, using convention W[k*n+i, r]
    """
    n = m = p = 6
    rank = 189
    
    # Initialize factor matrices
    U = np.zeros((36, rank), dtype=int)
    V = np.zeros((36, rank), dtype=int)
    W = np.zeros((36, rank), dtype=int)
    
    # Block layout for 6x6 split into 2x2 blocks of 3x3:
    # Block (br, bc) covers rows 3*br..3*br+2, cols 3*bc..3*bc+2
    # A blocks: A00=(0:3,0:3), A01=(0:3,3:6), A10=(3:6,0:3), A11=(3:6,3:6)
    # B blocks: B00=(0:3,0:3), B01=(0:3,3:6), B10=(3:6,0:3), B11=(3:6,3:6)
    # C blocks: C00=(0:3,0:3), C01=(0:3,3:6), C10=(3:6,0:3), C11=(3:6,3:6)
    
    # Strassen terms for 2x2 block multiplication:
    # T0: (A00+A11) * (B00+B11)
    # T1: (A10+A11) * B00
    # T2: A00 * (B01-B11)
    # T3: A11 * (B10-B00)
    # T4: (A00+A01) * B11
    # T5: (A10-A00) * (B00+B01)
    # T6: (A01-A11) * (B10+B11)
    
    # Each term: (left_block_combos, right_block_combos)
    # where combos map (block_row, block_col) -> coefficient
    strassen_terms = [
        # T0: (A00+A11)*(B00+B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # T1: (A10+A11)*B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),
        # T2: A00*(B01-B11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}),
        # T3: A11*(B10-B00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}),
        # T4: (A00+A01)*B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # T5: (A10-A00)*(B00+B01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
        # T6: (A01-A11)*(B10+B11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
    ]
    
    # Result block contributions: C_block = sum of signed T terms
    # C00 = T0 + T3 - T4 + T6
    # C01 = T2 + T4
    # C10 = T1 + T3
    # C11 = T0 - T1 + T2 + T5
    result_contribs = {
        (0,0): [(0, 1), (3, 1), (4, -1), (6, 1)],
        (0,1): [(2, 1), (4, 1)],
        (1,0): [(1, 1), (3, 1)],
        (1,1): [(0, 1), (1, -1), (2, 1), (5, 1)],
    }
    
    # For each Strassen term, decompose the 3x3 * 3x3 product via schoolbook.
    # Each scalar multiplication x_{i,a} * y_{a,k} becomes one rank-1 term.
    
    col = 0
    for t_idx, (left_combo, right_combo) in enumerate(strassen_terms):
        for i in range(3):  # row index within 3x3 block
            for a in range(3):  # middle index
                for k in range(3):  # col index within 3x3 block
                    # x_{i,a} = sum over (br,bc) in left_combo: coeff * A[3*br+i, 3*bc+a]
                    # y_{a,k} = sum over (br,bc) in right_combo: coeff * B[3*br+a, 3*bc+k]
                    
                    # Fill U: for each block in left_combo
                    for (br, bc), coeff_l in left_combo.items():
                        row_A = 3 * br + i
                        col_A = 3 * bc + a
                        U[row_A * 6 + col_A, col] += coeff_l
                    
                    # Fill V: for each block in right_combo
                    for (br, bc), coeff_r in right_combo.items():
                        row_B = 3 * br + a
                        col_B = 3 * bc + k
                        V[row_B * 6 + col_B, col] += coeff_r
                    
                    # Fill W: this term contributes to result blocks with signs
                    # C[i_total, k_total] where i_total = 3*rbr + i, k_total = 3*rbc + k
                    # W row index = k_total * n + i_total = (3*rbc+k)*6 + (3*rbr+i)
                    for (rbr, rbc), signs in result_contribs.items():
                        for (si, sign) in signs:
                            if si == t_idx:
                                i_total = 3 * rbr + i
                                k_total = 3 * rbc + k
                                W_row = k_total * 6 + i_total
                                W[W_row, col] += sign
                    
                    col += 1
    
    # Convert to lists for the required output format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
