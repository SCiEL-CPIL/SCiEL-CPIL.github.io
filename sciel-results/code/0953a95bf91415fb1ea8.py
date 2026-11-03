"""Program for matrix multiplication 4x4x5 tensor decomposition."""
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-65 exact tensor decomposition for multiplying a 4x4 matrix by a 4x5 matrix.
    
    Strategy:
    - Split into blocks: A is 4x4 (four 2x2 blocks), B is 4x5 (two 2x2 + one 2x1 block per row pair)
    - Use Strassen (rank 7) for the 4x4x4 part: 49 terms
    - Use schoolbook for the 4x4x1 part (5th column): 16 terms
    - Total rank: 65
    """
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    rank = 65
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65
    
    # Helper: add a rank-1 term with given coefficients
    def add_term(u_idx, v_idx, w_idx, cu, cv, cw, col):
        U[u_idx][col] += cu
        V[v_idx][col] += cv
        W[w_idx][col] += cw
    
    # Strassen decomposition for 2x2 matrix multiplication
    # For Z = X * Y where X, Y are 2x2:
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # 
    # m1 = (x11 - x00)(y10 + y11)
    # m2 = (x10 + x11)(y00)
    # m3 = (x00)(y00 - y11)
    # m4 = (x11)(y11 - y00)
    # m5 = (x00 + x11)(y00 + y11)
    # m6 = (x10 - x00)(y01 + y00)
    # m7 = (x01 + x11)(y10 - y11)
    #
    # Z00 = m1 - m4 + m5 + m7
    # Z01 = m3 + m6
    # Z10 = m2 + m5
    # Z11 = m1 - m3 + m4 + m6
    
    # Strassen basis vectors for 2x2 (flattened as [00, 01, 10, 11]):
    # Each mj = (linear_form_in_X) * (linear_form_in_Y)
    
    # For the block Strassen approach on 4x4x4:
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bij is 2x2
    # C = [[C00, C01], [C10, C11]] where each Cij is 2x2
    #
    # Apply Strassen at block level:
    # M1 = (A11 - A00)(B10 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B00 - B11)
    # M4 = A11(B11 - B00)
    # M5 = (A00 + A11)(B00 + B11)
    # M6 = (A10 - A00)(B01 + B00)
    # M7 = (A01 + A11)(B10 - B11)
    #
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M6
    # C10 = M2 + M5
    # C11 = M1 - M3 + M4 + M6
    #
    # Each Mj is a 2x2 matrix product, which we decompose using Strassen (7 terms each).
    # Total: 7 * 7 = 49 terms for the 4x4x4 part.
    
    # Let me define the Strassen decomposition for 2x2 as a set of (u_vec, v_vec, w_vec) triples
    # where u_vec is 4-dim (for flattened 2x2 X), v_vec is 4-dim (for flattened 2x2 Y),
    # w_vec is 4-dim (for flattened 2x2 Z), and Z = X*Y = sum_r w_r * (u_r . X) * (v_r . Y)
    
    # Strassen rank-7 decomposition for 2x2:
    # Indexing: 0=00, 1=01, 2=10, 3=11
    
    # m1 = (x11 - x00)(y10 + y11)
    # u1 = [-1, 0, 0, 1], v1 = [0, 0, 1, 1]
    # m2 = (x10 + x11)(y00)
    # u2 = [0, 0, 1, 1], v2 = [1, 0, 0, 0]
    # m3 = (x00)(y00 - y11)
    # u3 = [1, 0, 0, 0], v3 = [1, 0, 0, -1]
    # m4 = (x11)(y11 - y00)
    # u4 = [0, 0, 0, 1], v4 = [-1, 0, 0, 1]
    # m5 = (x00 + x11)(y00 + y11)
    # u5 = [1, 0, 0, 1], v5 = [1, 0, 0, 1]
    # m6 = (x10 - x00)(y01 + y00)
    # u6 = [-1, 0, 1, 0], v6 = [1, 1, 0, 0]
    # m7 = (x01 + x11)(y10 - y11)
    # u7 = [0, 1, 0, 1], v7 = [0, 0, 1, -1]
    
    strassen_u = [
        [-1, 0, 0, 1],  # m1
        [0, 0, 1, 1],   # m2
        [1, 0, 0, 0],   # m3
        [0, 0, 0, 1],   # m4
        [1, 0, 0, 1],   # m5
        [-1, 0, 1, 0],  # m6
        [0, 1, 0, 1],   # m7
    ]
    strassen_v = [
        [0, 0, 1, 1],   # m1
        [1, 0, 0, 0],   # m2
        [1, 0, 0, -1],  # m3
        [-1, 0, 0, 1],  # m4
        [1, 0, 0, 1],   # m5
        [1, 1, 0, 0],   # m6
        [0, 0, 1, -1],  # m7
    ]
    
    # Now for the output Z = X*Y, we need w vectors such that:
    # Z00 = m1 - m4 + m5 + m7
    # Z01 = m3 + m6
    # Z10 = m2 + m5
    # Z11 = m1 - m3 + m4 + m6
    #
    # So w for each output element:
    # w00 (for Z00): [1, 0, 0, -1, 1, 0, 1]  (coefficients for m1..m7)
    # w01 (for Z01): [0, 0, 1, 0, 0, 1, 0]
    # w10 (for Z10): [0, 1, 0, 0, 1, 0, 0]
    # w11 (for Z11): [1, 0, -1, 1, 0, 1, 0]
    
    strassen_w = [
        [1, 0, 0, -1, 1, 0, 1],   # Z00
        [0, 0, 1, 0, 0, 1, 0],    # Z01
        [0, 1, 0, 0, 1, 0, 0],    # Z10
        [1, 0, -1, 1, 0, 1, 0],   # Z11
    ]
    
    # For the block Strassen, we have 7 block-level multiplications M1..M7.
    # Each Mj = Xj * Yj where Xj and Yj are 2x2 matrices (linear combinations of A blocks).
    # We decompose each Xj * Yj using the 7-term Strassen decomposition above.
    # So we get 7*7 = 49 scalar rank-1 terms.
    
    # Block-level Strassen:
    # M1 = (A11 - A00)(B10 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B00 - B11)
    # M4 = A11(B11 - B00)
    # M5 = (A00 + A11)(B00 + B11)
    # M6 = (A10 - A00)(B01 + B00)
    # M7 = (A01 + A11)(B10 - B11)
    
    # For each Mj, we need to express Xj (the left 2x2 matrix) and Yj (the right 2x2 matrix)
    # as linear combinations of the 4 A-blocks and 4 B-blocks respectively.
    
    # A blocks: A00, A01, A10, A11 (each 2x2)
    # B blocks: B00, B01, B10, B11 (each 2x2)
    
    # Xj coefficients (which A blocks contribute, with what sign):
    # M1: X1 = A11 - A00  -> coeffs: [-1, 0, 0, 1] for [A00, A01, A10, A11]
    # M2: X2 = A10 + A11  -> coeffs: [0, 0, 1, 1]
    # M3: X3 = A00         -> coeffs: [1, 0, 0, 0]
    # M4: X4 = A11         -> coeffs: [0, 0, 0, 1]
    # M5: X5 = A00 + A11   -> coeffs: [1, 0, 0, 1]
    # M6: X6 = A10 - A00   -> coeffs: [-1, 0, 1, 0]
    # M7: X7 = A01 + A11   -> coeffs: [0, 1, 0, 1]
    
    block_X_coeffs = [
        [-1, 0, 0, 1],  # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 0, 0, 1],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, 1],   # M7
    ]
    
    # Yj coefficients (which B blocks contribute, with what sign):
    # M1: Y1 = B10 + B11  -> coeffs: [0, 0, 1, 1] for [B00, B01, B10, B11]
    # M2: Y2 = B00         -> coeffs: [1, 0, 0, 0]
    # M3: Y3 = B00 - B11   -> coeffs: [1, 0, 0, -1]
    # M4: Y4 = B11 - B00   -> coeffs: [-1, 0, 0, 1]
    # M5: Y5 = B00 + B11   -> coeffs: [1, 0, 0, 1]
    # M6: Y6 = B01 + B00   -> coeffs: [1, 1, 0, 0]
    # M7: Y7 = B10 - B11   -> coeffs: [0, 0, 1, -1]
    
    block_Y_coeffs = [
        [0, 0, 1, 1],   # M1
        [1, 0, 0, 0],   # M2
        [1, 0, 0, -1],  # M3
        [-1, 0, 0, 1],  # M4
        [1, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, -1],  # M7
    ]
    
    # Output C blocks: C00, C01, C10, C11
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M6
    # C10 = M2 + M5
    # C11 = M1 - M3 + M4 + M6
    
    block_C_coeffs = [
        [1, 0, 0, -1, 1, 0, 1],   # C00
        [0, 0, 1, 0, 0, 1, 0],    # C01
        [0, 1, 0, 0, 1, 0, 0],    # C10
        [1, 0, -1, 1, 0, 1, 0],   # C11
    ]
    
    # Now, for each block-level term Mj and each Strassen sub-term s (0..6),
    # we have a rank-1 term in the overall decomposition.
    
    # The U vector for this term:
    # For A block Ab (b in 0..3), the Strassen sub-term s uses u_s (4-dim vector for the 2x2 X).
    # The contribution to U from block Ab is: block_X_coeffs[j][b] * u_s
    # flattened into the 16-dim U space.
    
    # A is 4x4, flattened as A[i*4+j] where i is row, j is col.
    # Block A00 = rows 0-1, cols 0-1: indices 0,1,4,5
    # Block A01 = rows 0-1, cols 2-3: indices 2,3,6,7
    # Block A10 = rows 2-3, cols 0-1: indices 8,9,12,13
    # Block A11 = rows 2-3, cols 2-3: indices 10,11,14,15
    
    # For a 2x2 block with local indices [00, 01, 10, 11], the global indices are:
    # A00: [0, 1, 4, 5]
    # A01: [2, 3, 6, 7]
    # A10: [8, 9, 12, 13]
    # A11: [10, 11, 14, 15]
    
    A_block_indices = [
        [0, 1, 4, 5],   # A00
        [2, 3, 6, 7],   # A01
        [8, 9, 12, 13], # A10
        [10, 11, 14, 15] # A11
    ]
    
    # B is 4x5, flattened as B[j*5+k] where j is row, k is col.
    # For the 4x4x4 part, we only use cols 0-3 of B.
    # Block B00 = rows 0-1, cols 0-1: indices 0,1,5,6
    # Block B01 = rows 0-1, cols 2-3: indices 2,3,7,8
    # Block B10 = rows 2-3, cols 0-1: indices 10,11,15,16
    # Block B11 = rows 2-3, cols 2-3: indices 12,13,17,18
    
    B_block_indices_44 = [
        [0, 1, 5, 6],   # B00
        [2, 3, 7, 8],   # B01
        [10, 11, 15, 16], # B10
        [12, 13, 17, 18]  # B11
    ]
    
    # C is 4x5, flattened as C[k*4+i] where k is col, i is row (per task convention W[k*n+i,r])
    # For the 4x4x4 part, we use cols 0-3 of C.
    # Block C00 = rows 0-1, cols 0-1: C indices k*4+i for k in 0-1, i in 0-1: [0,1,4,5]
    # Block C01 = rows 0-1, cols 2-3: C indices k*4+i for k in 2-3, i in 0-1: [8,9,12,13]
    # Block C10 = rows 2-3, cols 0-1: C indices k*4+i for k in 0-1, i in 2-3: [2,3,6,7]
    # Block C11 = rows 2-3, cols 2-3: C indices k*4+i for k in 2-3, i in 2-3: [10,11,14,15]
    
    C_block_indices_44 = [
        [0, 1, 4, 5],   # C00 (rows 0-1, cols 0-1)
        [8, 9, 12, 13], # C01 (rows 0-1, cols 2-3)
        [2, 3, 6, 7],   # C10 (rows 2-3, cols 0-1)
        [10, 11, 14, 15] # C11 (rows 2-3, cols 2-3)
    ]
    
    # Build the 49 terms for the 4x4x4 part
    col = 0
    for j in range(7):  # block-level Strassen term Mj
        for s in range(7):  # sub-level Strassen term
            # U contribution
            for b in range(4):  # A block
                coeff_x = block_X_coeffs[j][b]
                if coeff_x == 0:
                    continue
                for loc in range(4):
                    u_coeff = strassen_u[s][loc]
                    if u_coeff == 0:
                        continue
                    u_idx = A_block_indices[b][loc]
                    U[u_idx][col] += coeff_x * u_coeff
            
            # V contribution
            for b in range(4):  # B block
                coeff_y = block_Y_coeffs[j][b]
                if coeff_y == 0:
                    continue
                for loc in range(4):
                    v_coeff = strassen_v[s][loc]
                    if v_coeff == 0:
                        continue
                    v_idx = B_block_indices_44[b][loc]
                    V[v_idx][col] += coeff_y * v_coeff
            
            # W contribution
            for c in range(4):  # C block
                coeff_c = block_C_coeffs[c][j]
                if coeff_c == 0:
                    continue
                for loc in range(4):
                    w_coeff = strassen_w[loc][s]
                    if w_coeff == 0:
                        continue
                    w_idx = C_block_indices_44[c][loc]
                    W[w_idx][col] += coeff_c * w_coeff
            
            col += 1
    
    # Now for the 4x4x1 part (5th column, k=4)
    # C[i,4] = sum_{j=0}^{3} A[i,j] * B[j,4]
    # This is schoolbook: 16 terms
    # U[i*4+j][col] = 1, V[j*5+4][col] = 1, W[4*4+i][col] = 1
    
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1  # j*5 + 4
            W[4 * n + i][col] = 1  # 4*4 + i = 16 + i
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
