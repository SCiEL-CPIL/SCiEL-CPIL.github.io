import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    # Strassen coefficients for 2x2x2x2
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    alpha = [
        [1, 0, 1, 1],  # M1: A00 + A11
        [0, 1, 0, 1],  # M2: A10 + A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: A00 + A01
        [-1, 0, 1, 0], # M6: A10 - A00
        [0, 1, 0, -1], # M7: A01 - A11
    ]
    beta = [
        [1, 1, 0, 0],  # M1: B00 + B11
        [1, 0, 0, 0],  # M2: B00
        [0, 1, 0, -1], # M3: B01 - B11
        [0, -1, 1, 0], # M4: B10 - B00
        [0, 0, 0, 1],  # M5: B11
        [1, 1, 0, 0],  # M6: B00 + B01
        [0, 0, 1, 1],  # M7: B10 + B11
    ]
    # recon[i] = list of (block_row, block_col, coeff) for output C
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    recon = [
        [(0, 0, 1), (1, 1, 1)],      # M1 -> C00, C11
        [(1, 0, 1)],                  # M2 -> C10
        [(0, 1, 1)],                  # M3 -> C01
        [(0, 0, 1), (1, 0, 1)],      # M4 -> C00, C10
        [(0, 0, -1), (0, 1, 1)],     # M5 -> C00(-1), C01(+1)
        [(1, 1, 1)],                  # M6 -> C11
        [(0, 0, 1), (1, 1, 1)],      # M7 -> C00, C11
    ]
    
    col = 0
    
    # Part 1: A * B1 (4x4 * 4x4) using Strassen recursion
    # 7x7 = 49 terms
    for s in range(7):
        for si in range(7):
            # U: A coefficients
            # A is 4x4, indexed as A[r,c] where r,c in 0..3
            # Block structure: A[br, bc] is 2x2 block at row br, col bc
            # Global index: (br*2 + r_local)*4 + (bc*2 + c_local)
            for a in range(4):
                if alpha[si][a] != 0:
                    a_br, a_bc = a // 2, a % 2
                    for r in range(4):
                        if alpha[s][r] != 0:
                            r_br, r_bc = r // 2, r % 2
                            # U index: global A index
                            # The product is (sum_r alpha[s][r] * A_block_r) * (sum_a alpha[si][a] * A_block_a)
                            # Wait, this is wrong. Let me reconsider.
                            # In Strassen for 4x4, we have 7 "super-products"
                            # Each super-product is a 2x2 matrix product
                            # So we need 7 Strassen products at the 4x4 level
                            # Each of those uses 7 Strassen products at the 2x2 level
                            # Total 49
                            # U[i, col] = coefficient for A[i]
                            # The structure is: for each of the 7 outer Strassen products (s),
                            # we form a linear combination of A's 2x2 blocks
                            # Then for each of the 7 inner Strassen products (si),
                            # we form a linear combination of the 2x2 sub-blocks
                            # This is getting complex. Let me use a different approach.
                            pass
            
            # This approach is getting too complex. Let me use a precomputed rank-49 decomposition.
            pass
    
    # Let me use a simpler, verified approach.
    # I'll construct the decomposition using the known rank-49 for 4x4x4x4.
    
    # Actually, let me just fix the bugs in the original approach more carefully.
    
    # Reset
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    col = 0
    
    # For 4x4x4x4, use the standard Strassen-based rank-49 decomposition.
    # The 7 outer products correspond to the 7 Strassen products for 4x4.
    # Each outer product s has coefficients for the 4 2x2 blocks of A and B.
    # Then each outer product is itself a 2x2x2x2 Strassen product with 7 inner products.
    
    # Outer Strassen for 4x4:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Outer alpha (for A blocks):
    outer_alpha = [
        [1, 0, 1, 1],  # M1: A00 + A11
        [0, 1, 0, 1],  # M2: A10 + A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: A00 + A01
        [-1, 0, 1, 0], # M6: A10 - A00
        [0, 1, 0, -1], # M7: A01 - A11
    ]
    outer_beta = [
        [1, 1, 0, 0],  # M1: B00 + B11
        [1, 0, 0, 0],  # M2: B00
        [0, 1, 0, -1], # M3: B01 - B11
        [0, -1, 1, 0], # M4: B10 - B00
        [0, 0, 0, 1],  # M5: B11
        [1, 1, 0, 0],  # M6: B00 + B01
        [0, 0, 1, 1],  # M7: B10 + B11
    ]
    outer_recon = [
        [(0, 0, 1), (1, 1, 1)],      # M1
        [(1, 0, 1)],                  # M2
        [(0, 1, 1)],                  # M3
        [(0, 0, 1), (1, 0, 1)],      # M4
        [(0, 0, -1), (0, 1, 1)],     # M5
        [(1, 1, 1)],                  # M6
        [(0, 0, 1), (1, 1, 1)],      # M7
    ]
    
    # Inner Strassen for 2x2:
    inner_alpha = alpha
    inner_beta = beta
    inner_recon = recon
    
    for s in range(7):
        for si in range(7):
            # U: A coefficients
            for a in range(4):
                if outer_alpha[s][a] != 0:
                    a_br, a_bc = a // 2, a % 2
                    for r in range(4):
                        if inner_alpha[si][r] != 0:
                            r_br, r_bc = r // 2, r % 2
                            u_idx = (a_br * 2 + r_br) * 4 + (a_bc * 2 + r_bc)
                            U[u_idx, col] += outer_alpha[s][a] * inner_alpha[si][r]
            
            # V: B1 coefficients
            for b in range(4):
                if outer_beta[s][b] != 0:
                    b_br, b_bc = b // 2, b % 2
                    for c in range(4):
                        if inner_beta[si][c] != 0:
                            c_br, c_bc = c // 2, c % 2
                            v_idx = (b_br * 2 + c_br) * 6 + (b_bc * 2 + c_bc)
                            V[v_idx, col] += outer_beta[s][b] * inner_beta[si][c]
            
            # W: Output C1 coefficients
            for a_blk, b_blk, gamma_outer in outer_recon[s]:
                for za, zb, gamma_inner in inner_recon[si]:
                    w_idx = (b_blk * 2 + zb) * 4 + (a_blk * 2 + za)
                    W[w_idx, col] += gamma_outer * gamma_inner
            
            col += 1
    
    # Part 2: A * B2 (4x4 * 4x2)
    # B2 has columns 4-5
    # Split into 4 terms: A00*B20, A01*B21, A10*B20, A11*B21
    # Each is 2x2 * 2x2, rank 7
    
    terms = [
        (0, 0, 0),  # A_block, B2_block, C_row_block
        (1, 1, 0),
        (2, 0, 1),
        (3, 1, 1)
    ]
    
    for a_blk, b_blk, c_r in terms:
        ar_blk, ac_blk = a_blk // 2, a_blk % 2
        for si in range(7):
            # U
            for a in range(4):
                if inner_alpha[si][a] != 0:
                    ar, ac = a // 2, a % 2
                    u_idx = (ar_blk * 2 + ar) * 4 + (ac_blk * 2 + ac)
                    U[u_idx, col] += inner_alpha[si][a]
            
            # V
            # B2 blocks: B20 (rows 0-1, cols 4-5), B21 (rows 2-3, cols 4-5)
            br_blk = b_blk
            for b in range(4):
                if inner_beta[si][b] != 0:
                    br, bc = b // 2, b % 2
                    v_idx = (br_blk * 2 + br) * 6 + (4 + bc)
                    V[v_idx, col] += inner_beta[si][b]
            
            # W
            for za, zb, zgamma in inner_recon[si]:
                w_idx = (4 + zb) * 4 + (c_r * 2 + za)
                W[w_idx, col] += zgamma
            
            col += 1
    
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
