import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # Total rank is 114
    rank = 114
    U = [[0] * rank for _ in range(n * m)]  # 16 x 114
    V = [[0] * rank for _ in range(m * p)]  # 36 x 114
    W = [[0] * rank for _ in range(p * n)]  # 36 x 114
    
    # Helper to add a term to U, V, W
    # u_idx, u_val, v_idx, v_val, w_idx, w_val
    def add_term(col, u_idx, u_val, v_idx, v_val, w_idx, w_val):
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val

    # 2x2x2 Strassen Decomposition
    # Returns list of (u_vec, v_vec, w_vec) for the 2x2 case
    # u_vec, v_vec, w_vec are length 4
    def strassen_2x2():
        terms = []
        # P1 = (a00+a11)(b00+b11)
        u1 = [1, 0, 0, 1]
        v1 = [1, 0, 0, 1]
        w1 = [1, 0, 0, 1] # c00, c01, c10, c11
        terms.append((u1, v1, w1))
        
        # P2 = (a10+a11)b00
        u2 = [0, 0, 1, 1]
        v2 = [1, 0, 0, 0]
        w2 = [0, 0, 1, -1]
        terms.append((u2, v2, w2))
        
        # P3 = a00(b01-b11)
        u3 = [1, 0, 0, 0]
        v3 = [0, 1, 0, -1]
        w3 = [0, 1, 0, 1]
        terms.append((u3, v3, w3))
        
        # P4 = a11(b10-b00)
        u4 = [0, 0, 0, 1]
        v4 = [0, 0, 1, -1] # b10 is index 2, b00 is index 0. Wait.
        # B indices: 00(0), 01(1), 10(2), 11(3)
        # v4 should be b10 - b00 -> [ -1, 0, 1, 0 ]
        v4 = [-1, 0, 1, 0]
        w4 = [1, 0, 1, 0] # c00, c10
        terms.append((u4, v4, w4))
        
        # P5 = (a00+a01)b11
        u5 = [1, 1, 0, 0]
        v5 = [0, 0, 0, 1]
        w5 = [-1, 1, 0, 0] # c00, c01
        terms.append((u5, v5, w5))
        
        # P6 = (a10-a00)(b00+b01)
        u6 = [-1, 0, 1, 0]
        v6 = [1, 1, 0, 0]
        w6 = [0, 0, 0, 1] # c11
        terms.append((u6, v6, w6))
        
        # P7 = (a01-a11)(b10+b11)
        u7 = [0, 1, 0, -1]
        v7 = [0, 0, 1, 1]
        w7 = [1, 0, 0, 1] # c00, c11
        terms.append((u7, v7, w7))
        
        return terms

    strassen_terms = strassen_2x2()
    
    # Macro terms for 4x4x4
    # Each macro term is defined by:
    # - A blocks involved and coefficients
    # - B blocks involved and coefficients
    # - C blocks involved and coefficients
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    # C blocks: 00, 01, 10, 11
    
    # We map block indices to row/col offsets in the 4x4 matrix
    # A block (r,c) occupies rows 2r..2r+1, cols 2c..2c+1
    # A index = i * 4 + j
    
    def block_indices(block):
        r, c = block
        return [(2*r + i) * 4 + (2*c + j) for i in range(2) for j in range(2)]

    # List of (A_blocks, B_blocks, C_blocks)
    # A_blocks: list of (block, coeff)
    # B_blocks: list of (block, coeff)
    # C_blocks: list of (block, coeff)
    
    macro_terms = [
        # 1
        {
            'A': [((0,0), 1), ((1,1), 1)],
            'B': [((0,0), 1), ((1,1), 1)],
            'C': [((0,0), 1), ((1,1), 1)]
        },
        # 2
        {
            'A': [((1,0), 1), ((1,1), 1)],
            'B': [((0,0), 1)],
            'C': [((1,0), 1), ((1,1), -1)]
        },
        # 3
        {
            'A': [((0,0), 1)],
            'B': [((0,1), 1), ((1,1), -1)],
            'C': [((0,1), 1), ((1,1), 1)]
        },
        # 4
        {
            'A': [((1,1), 1)],
            'B': [((1,0), 1), ((0,0), -1)],
            'C': [((0,0), 1), ((1,0), 1)]
        },
        # 5
        {
            'A': [((0,0), 1), ((0,1), 1)],
            'B': [((1,1), 1)],
            'C': [((0,0), -1), ((0,1), 1)]
        },
        # 6
        {
            'A': [((1,0), 1), ((0,0), -1)],
            'B': [((0,0), 1), ((0,1), 1)],
            'C': [((1,1), 1)]
        },
        # 7
        {
            'A': [((0,1), 1), ((1,1), -1)],
            'B': [((1,0), 1), ((1,1), 1)],
            'C': [((0,0), 1), ((1,1), 1)]
        }
    ]

    col = 0
    for macro in macro_terms:
        # Construct linear forms for A and B blocks
        # We need to combine the 2x2 Strassen terms with the block linear forms
        # For each 2x2 Strassen term (u, v, w):
        # u is a vector of length 4 for the 2x2 A block
        # v is a vector of length 4 for the 2x2 B block
        # w is a vector of length 4 for the 2x2 C block
        
        # The macro A is a sum of blocks.
        # Let A_macro = sum_k coeff_k * A_block_k
        # The 2x2 Strassen term u applies to A_macro.
        # So the effective U vector for the 4x4 case is:
        # sum_k coeff_k * (u expanded to 4x4 block k)
        
        # Similarly for V.
        
        # The macro C is a sum of blocks.
        # C_macro = sum_l coeff_l * C_block_l
        # The 2x2 Strassen term w applies to C_macro.
        # So the effective W vector for the 4x4 case is:
        # sum_l coeff_l * (w expanded to 4x4 block l)
        
        for u_2x2, v_2x2, w_2x2 in strassen_terms:
            # Build U vector
            U_vec = [0] * 16
            for (block, coeff) in macro['A']:
                indices = block_indices(block)
                for idx, u_val in zip(indices, u_2x2):
                    U_vec[idx] += coeff * u_val
            
            # Build V vector
            V_vec = [0] * 16
            for (block, coeff) in macro['B']:
                indices = block_indices(block)
                for idx, v_val in zip(indices, v_2x2):
                    V_vec[idx] += coeff * v_val
            
            # Build W vector
            W_vec = [0] * 16
            for (block, coeff) in macro['C']:
                indices = block_indices(block)
                for idx, w_val in zip(indices, w_2x2):
                    W_vec[idx] += coeff * w_val
            
            # Add to global U, V, W
            # Note: V in global is m*p = 4*9 = 36.
            # Here we are handling the first 4 columns of B (4x4 part).
            # So V indices 0..15 correspond to B[0:4, 0:4].
            # Wait, B is 4x9.
            # B indices: row 0..3, col 0..8.
            # V index = row * 9 + col.
            # For the first 4 columns (col 0..3), V indices are row*9 + col.
            
            for i in range(16):
                if U_vec[i] != 0:
                    U[i][col] = U_vec[i]
            
            for i in range(16):
                if V_vec[i] != 0:
                    # Map 4x4 block index to 4x9 matrix index
                    # i is index in 4x4 (row*4+col)
                    r = i // 4
                    c = i % 4
                    v_idx = r * 9 + c
                    V[v_idx][col] = V_vec[i]
            
            for i in range(16):
                if W_vec[i] != 0:
                    # Map 4x4 block index to 4x9 output index
                    # Output is 4x9.
                    # W index = row * 4 + col?
                    # Task says W[k*n+i, r].
                    # W is p*n. p=9, n=4.
                    # Output C is 4x9? No, C is A(4x4) * B(4x9) = 4x9.
                    # So C has 4 rows and 9 cols.
                    # W index = k * n + i.
                    # k is col of C (0..8), i is row of C (0..3).
                    # Wait, "Output factor follows the official trace convention W[k*n+i,r]".
                    # Usually C[i,j] = sum_k A[i,k] B[k,j].
                    # Here A is 4x4, B is 4x9.
                    # C[i,j] where i in 0..3, j in 0..8.
                    # W index = j * 4 + i.
                    
                    # In our 4x4x4 decomposition, we computed C_block.
                    # The 4x4 result corresponds to C[0:4, 0:4].
                    # So for the first 4x4 part, j is 0..3.
                    # W index = j * 4 + i.
                    
                    r = i // 4
                    c = i % 4
                    w_idx = c * 4 + r
                    W[w_idx][col] = W_vec[i]
            
            col += 1
    
    # Now handle the second 4x4 part (columns 4..7 of B)
    # This is identical to the first, but B indices are shifted.
    # B cols 4..7 correspond to v_idx = r * 9 + (4 + c).
    # C cols 4..7 correspond to w_idx = (4 + c) * 4 + r.
    
    for macro in macro_terms:
        for u_2x2, v_2x2, w_2x2 in strassen_terms:
            U_vec = [0] * 16
            for (block, coeff) in macro['A']:
                indices = block_indices(block)
                for idx, u_val in zip(indices, u_2x2):
                    U_vec[idx] += coeff * u_val
            
            V_vec = [0] * 16
            for (block, coeff) in macro['B']:
                indices = block_indices(block)
                for idx, v_val in zip(indices, v_2x2):
                    V_vec[idx] += coeff * v_val
            
            W_vec = [0] * 16
            for (block, coeff) in macro['C']:
                indices = block_indices(block)
                for idx, w_val in zip(indices, w_2x2):
                    W_vec[idx] += coeff * w_val
            
            for i in range(16):
                if U_vec[i] != 0:
                    U[i][col] = U_vec[i]
            
            for i in range(16):
                if V_vec[i] != 0:
                    r = i // 4
                    c = i % 4
                    v_idx = r * 9 + (4 + c)
                    V[v_idx][col] = V_vec[i]
            
            for i in range(16):
                if W_vec[i] != 0:
                    r = i // 4
                    c = i % 4
                    w_idx = (4 + c) * 4 + r
                    W[w_idx][col] = W_vec[i]
            
            col += 1
    
    # Now handle the 4x4x1 part (column 8 of B)
    # Rank 16.
    # C_k = sum_j A_{kj} B_j.
    # B is 4x1. Only column 8.
    # B index = r * 9 + 8.
    # C index = 8 * 4 + k.
    
    for k in range(4):
        for j in range(4):
            # A index = k * 4 + j
            u_idx = k * 4 + j
            # B index = j * 9 + 8
            v_idx = j * 9 + 8
            # W index = 8 * 4 + k
            w_idx = 8 * 4 + k
            
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
