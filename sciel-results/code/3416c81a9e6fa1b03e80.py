import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    bs = 3  # block size
    
    # We use Strassen's algorithm at the 2x2 block level (each entry is a 3x3 matrix).
    # The 7 Strassen products are:
    # P1 = (A00 + A11) * (B00 + B11)
    # P2 = (A10 + A11) * B00
    # P3 = A00 * (B01 - B11)
    # P4 = A11 * (B10 - B00)
    # P5 = (A00 + A01) * B11
    # P6 = (A10 - A00) * (B00 + B01)
    # P7 = (A01 - A11) * (B10 + B11)
    
    # Each P is a 3x3 matrix product. We use the schoolbook rank-9 decomposition for each 3x3 product.
    # Total rank = 7 * 9 = 63.
    
    rank = 63
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Block offsets: (row_offset, col_offset) for each of the 4 blocks
    # Block 0: top-left (0,0)
    # Block 1: top-right (0,3)
    # Block 2: bottom-left (3,0)
    # Block 3: bottom-right (3,3)
    block_offsets = [
        (0, 0),  # 0: TL
        (0, 3),  # 1: TR
        (3, 0),  # 2: BL
        (3, 3)   # 3: BR
    ]
    
    # Strassen terms: each term is (alpha_dict, beta_dict, gamma_dict)
    # alpha: {block_idx: coeff} for A blocks
    # beta: {block_idx: coeff} for B blocks
    # gamma: {block_idx: coeff} for C blocks (contribution to output)
    
    # P1 = (A00 + A11)(B00 + B11)
    # C00 += P1, C11 += P1
    # P2 = (A10 + A11)B00
    # C10 += P2, C11 -= P2
    # P3 = A00(B01 - B11)
    # C01 += P3
    # P4 = A11(B10 - B00)
    # C00 += P4
    # P5 = (A00 + A01)B11
    # C00 -= P5, C01 += P5
    # P6 = (A10 - A00)(B00 + B01)
    # C11 += P6
    # P7 = (A01 - A11)(B10 + B11)
    # C00 += P7
    
    terms = [
        # P1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # P2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # P3
        ({0: 1}, {1: 1, 3: -1}, {1: 1}),
        # P4
        ({3: 1}, {2: 1, 0: -1}, {0: 1}),
        # P5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # P6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # P7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    col = 0
    for alpha, beta, gamma in terms:
        # For each 3x3 schoolbook term (i, k, j) where i, k, j in 0..2
        for i in range(3):
            for k in range(3):
                for j in range(3):
                    # U picks up X[i, k] where X is the combined A matrix
                    # X[i, k] = sum_{bA} alpha[bA] * A_{bA}[i, k]
                    for bA, coeff_a in alpha.items():
                        rA, cA = block_offsets[bA]
                        u_row = rA + i
                        u_col = cA + k
                        u_idx = u_row * m + u_col
                        U[u_idx, col] += coeff_a
                    
                    # V picks up Y[k, j] where Y is the combined B matrix
                    # Y[k, j] = sum_{bB} beta[bB] * B_{bB}[k, j]
                    for bB, coeff_b in beta.items():
                        rB, cB = block_offsets[bB]
                        v_row = rB + k
                        v_col = cB + j
                        v_idx = v_row * p + v_col
                        V[v_idx, col] += coeff_b
                    
                    # W accumulates Z[i, j] into C
                    # Z[i, j] contributes to C_{bC}[i, j] with coeff gamma[bC]
                    # W index is k*n + i (column k, row i of C)
                    for bC, coeff_c in gamma.items():
                        rC, cC = block_offsets[bC]
                        w_row_global = rC + i
                        w_col_global = cC + j
                        w_idx = w_col_global * n + w_row_global
                        W[w_idx, col] += coeff_c
                    
                    col += 1
    
    # Convert to lists as expected by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
