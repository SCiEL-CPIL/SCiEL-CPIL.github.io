import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    # U: (n*m, rank) = (16, rank)
    # V: (m*p, rank) = (20, rank)
    # W: (p*n, rank) = (20, rank)
    
    rank = 65
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((20, rank), dtype=int)
    W = np.zeros((20, rank), dtype=int)
    
    # --- Part 1: 4x4x4 Strassen (Rank 49) ---
    # Columns 0-3 of B and C
    
    # Helper to add a product term
    # u_coeffs: dict mapping A_index (0-15) to coeff
    # v_coeffs: dict mapping B_index (0-15) to coeff
    # w_coeffs: dict mapping C_index (0-15) to coeff
    def add_term(u_coeffs, v_coeffs, w_coeffs, col):
        for idx, val in u_coeffs.items():
            if val != 0:
                U[idx, col] += val
        for idx, val in v_coeffs.items():
            if val != 0:
                V[idx, col] += val
        for idx, val in w_coeffs.items():
            if val != 0:
                W[idx, col] += val

    # A indices: i*4 + j
    # B indices: j*5 + k (but we only use k=0..3, so j*5 + k)
    # C indices: k*4 + i
    
    # Block definitions for A (4x4)
    # A00: i in 0,1; j in 0,1 -> indices 0, 1, 4, 5
    # A01: i in 0,1; j in 2,3 -> indices 2, 3, 6, 7
    # A10: i in 2,3; j in 0,1 -> indices 8, 9, 12, 13
    # A11: i in 2,3; j in 2,3 -> indices 10, 11, 14, 15
    
    # B indices (cols 0-3): j*5 + k
    # B00: j in 0,1; k in 0,1 -> indices 0, 1, 5, 6
    # B01: j in 0,1; k in 2,3 -> indices 2, 3, 7, 8
    # B10: j in 2,3; k in 0,1 -> indices 10, 11, 15, 16
    # B11: j in 2,3; k in 2,3 -> indices 12, 13, 17, 18
    
    # C indices (cols 0-3): k*4 + i
    # C00: k in 0,1; i in 0,1 -> indices 0, 1, 4, 5
    # C01: k in 0,1; i in 2,3 -> indices 2, 3, 6, 7
    # C10: k in 2,3; i in 0,1 -> indices 8, 9, 12, 13
    # C11: k in 2,3; i in 2,3 -> indices 10, 11, 14, 15

    # Strassen Block Combinations
    # b=0: X = A10+A11, Y = B00+B01
    # b=1: X = A10+A11, Y = B00
    # b=2: X = A00,      Y = B10-B11
    # b=3: X = A11,      Y = B11
    # b=4: X = A00+A11,  Y = B00+B11
    # b=5: X = A01-A11,  Y = B10+B11
    # b=6: X = A01-A00,  Y = B00+B10

    # We need to represent X and Y as 2x2 matrices of linear forms over A and B.
    # Then apply the 7 scalar Strassen formulas.
    
    # Let's define the blocks as lists of (index, coeff) for A and B.
    # A blocks
    A00 = [(0, 1), (1, 1), (4, 1), (5, 1)]
    A01 = [(2, 1), (3, 1), (6, 1), (7, 1)]
    A10 = [(8, 1), (9, 1), (12, 1), (13, 1)]
    A11 = [(10, 1), (11, 1), (14, 1), (15, 1)]
    
    # B blocks (cols 0-3)
    B00 = [(0, 1), (1, 1), (5, 1), (6, 1)]
    B01 = [(2, 1), (3, 1), (7, 1), (8, 1)]
    B10 = [(10, 1), (11, 1), (15, 1), (16, 1)]
    B11 = [(12, 1), (13, 1), (17, 1), (18, 1)]

    # Combine blocks
    def combine_blocks(block1, block2, sign2=1):
        res = dict(block1)
        for idx, coeff in block2:
            res[idx] = res.get(idx, 0) + sign2 * coeff
        return res

    X_blocks = []
    Y_blocks = []
    
    # b=0
    X_blocks.append(combine_blocks(A10, A11))
    Y_blocks.append(combine_blocks(B00, B01))
    # b=1
    X_blocks.append(combine_blocks(A10, A11))
    Y_blocks.append(combine_blocks(B00))
    # b=2
    X_blocks.append(combine_blocks(A00))
    Y_blocks.append(combine_blocks(B10, B11, -1))
    # b=3
    X_blocks.append(combine_blocks(A11))
    Y_blocks.append(combine_blocks(B11))
    # b=4
    X_blocks.append(combine_blocks(A00, A11))
    Y_blocks.append(combine_blocks(B00, B11))
    # b=5
    X_blocks.append(combine_blocks(A01, A11, -1))
    Y_blocks.append(combine_blocks(B10, B11))
    # b=6
    X_blocks.append(combine_blocks(A01, A00, -1))
    Y_blocks.append(combine_blocks(B00, B10))

    # Scalar Strassen Products for 2x2
    # X = [[x00, x01], [x10, x11]]
    # Y = [[y00, y01], [y10, y11]]
    # S0 = (x10+x11)(y00+y01)
    # S1 = (x10+x11)y00
    # S2 = x00(y10-y11)
    # S3 = x11y11
    # S4 = (x00+x11)(y00+y11)
    # S5 = (x01-x11)(y10+y11)
    # S6 = (x01-x00)(y00+y10)
    
    # C = XY
    # C00 = S4+S3-S1+S5
    # C01 = S0-S4+S6
    # C10 = S2+S3
    # C11 = S0-S2+S3-S1

    # We need to extract x00, x01, x10, x11 from the combined A block.
    # The block is 2x2.
    # A block indices in 2x2:
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    # Global A index mapping for a block:
    # If block is A00 (i in 0,1; j in 0,1):
    #   0->0, 1->1, 2->4, 3->5
    # If block is A01 (i in 0,1; j in 2,3):
    #   0->2, 1->3, 2->6, 3->7
    # If block is A10 (i in 2,3; j in 0,1):
    #   0->8, 1->9, 2->12, 3->13
    # If block is A11 (i in 2,3; j in 2,3):
    #   0->10, 1->11, 2->14, 3->15
    
    # This mapping is complex because X_blocks are sums of different A blocks.
    # Instead of extracting, we can just sum the linear forms directly.
    
    # For each b, X_b is a linear form over A.
    # x00 is the (0,0) element of X_b.
    # We need to know which A indices contribute to x00, x01, x10, x11.
    
    # Let's re-construct X_b as 4 linear forms (x00, x01, x10, x11).
    # Each linear form is a dict of A_index -> coeff.
    
    def get_block_elements(block_dict, block_type):
        # block_type: 0=A00, 1=A01, 2=A10, 3=A11
        # Returns 4 dicts for x00, x01, x10, x11
        # This function is not directly usable because X_b is a sum of blocks.
        pass

    # Alternative: Expand the sums.
    # X_b is a sum of A_blocks.
    # Each A_block is a 2x2 matrix.
    # So X_b is a 2x2 matrix of linear forms.
    
    def add_2x2_blocks(block1, block2, sign2=1):
        # block1, block2 are dicts of A_index->coeff
        # We need to know the position of each index in the 2x2 block.
        # This is hard with just dicts.
        
        # Let's use the explicit index lists.
        # A00: [0, 1, 4, 5]
        # A01: [2, 3, 6, 7]
        # A10: [8, 9, 12, 13]
        # A11: [10, 11, 14, 15]
        
        # We will create a function that takes a list of (block_name, sign) and returns
        # 4 dicts: x00, x01, x10, x11.
        
        block_indices = {
            'A00': [0, 1, 4, 5],
            'A01': [2, 3, 6, 7],
            'A10': [8, 9, 12, 13],
            'A11': [10, 11, 14, 15]
        }
        
        x00 = {}; x01 = {}; x10 = {}; x11 = {}
        
        for b_name, sign in block1:
            idxs = block_indices[b_name]
            x00[idxs[0]] = x00.get(idxs[0], 0) + sign
            x01[idxs[1]] = x01.get(idxs[1], 0) + sign
            x10[idxs[2]] = x10.get(idxs[2], 0) + sign
            x11[idxs[3]] = x11.get(idxs[3], 0) + sign
            
        for b_name, sign in block2:
            idxs = block_indices[b_name]
            x00[idxs[0]] = x00.get(idxs[0], 0) + sign
            x01[idxs[1]] = x01.get(idxs[1], 0) + sign
            x10[idxs[2]] = x10.get(idxs[2], 0) + sign
            x11[idxs[3]] = x11.get(idxs[3], 0) + sign
            
        return x00, x01, x10, x11

    def get_y_elements(block1, block2, sign2=1):
        block_indices = {
            'B00': [0, 1, 5, 6],
            'B01': [2, 3, 7, 8],
            'B10': [10, 11, 15, 16],
            'B11': [12, 13, 17, 18]
        }
        y00 = {}; y01 = {}; y10 = {}; y11 = {}
        
        for b_name, sign in block1:
            idxs = block_indices[b_name]
            y00[idxs[0]] = y00.get(idxs[0], 0) + sign
            y01[idxs[1]] = y01.get(idxs[1], 0) + sign
            y10[idxs[2]] = y10.get(idxs[2], 0) + sign
            y11[idxs[3]] = y11.get(idxs[3], 0) + sign
            
        for b_name, sign in block2:
            idxs = block_indices[b_name]
            y00[idxs[0]] = y00.get(idxs[0], 0) + sign * sign2
            y01[idxs[1]] = y01.get(idxs[1], 0) + sign * sign2
            y10[idxs[2]] = y10.get(idxs[2], 0) + sign * sign2
            y11[idxs[3]] = y11.get(idxs[3], 0) + sign * sign2
            
        return y00, y01, y10, y11

    # Define block combinations as lists of (name, sign)
    X_defs = [
        [('A10', 1), ('A11', 1)],
        [('A10', 1), ('A11', 1)],
        [('A00', 1)],
        [('A11', 1)],
        [('A00', 1), ('A11', 1)],
        [('A01', 1), ('A11', -1)],
        [('A01', 1), ('A00', -1)]
    ]
    
    Y_defs = [
        [('B00', 1), ('B01', 1)],
        [('B00', 1)],
        [('B10', 1), ('B11', -1)],
        [('B11', 1)],
        [('B00', 1), ('B11', 1)],
        [('B10', 1), ('B11', 1)],
        [('B00', 1), ('B10', 1)]
    ]
    
    col = 0
    for b in range(7):
        x00, x01, x10, x11 = get_block_elements(X_defs[b][0], X_defs[b][1] if len(X_defs[b])>1 else [])
        # Wait, get_block_elements takes two lists. I defined it to take block1, block2.
        # My X_defs are lists of tuples. I need to adjust the call.
        
        # Let's refactor get_block_elements to take a list of tuples.
        
        def get_x_elements(defs):
            block_indices = {
                'A00': [0, 1, 4, 5],
                'A01': [2, 3, 6, 7],
                'A10': [8, 9, 12, 13],
                'A11': [10, 11, 14, 15]
            }
            x00 = {}; x01 = {}; x10 = {}; x11 = {}
            for b_name, sign in defs:
                idxs = block_indices[b_name]
                x00[idxs[0]] = x00.get(idxs[0], 0) + sign
                x01[idxs[1]] = x01.get(idxs[1], 0) + sign
                x10[idxs[2]] = x10.get(idxs[2], 0) + sign
                x11[idxs[3]] = x11.get(idxs[3], 0) + sign
            return x00, x01, x10, x11

        def get_y_elements(defs):
            block_indices = {
                'B00': [0, 1, 5, 6],
                'B01': [2, 3, 7, 8],
                'B10': [10, 11, 15, 16],
                'B11': [12, 13, 17, 18]
            }
            y00 = {}; y01 = {}; y10 = {}; y11 = {}
            for b_name, sign in defs:
                idxs = block_indices[b_name]
                y00[idxs[0]] = y00.get(idxs[0], 0) + sign
                y01[idxs[1]] = y01.get(idxs[1], 0) + sign
                y10[idxs[2]] = y10.get(idxs[2], 0) + sign
                y11[idxs[3]] = y11.get(idxs[3], 0) + sign
            return y00, y01, y10, y11

        x00, x01, x10, x11 = get_x_elements(X_defs[b])
        y00, y01, y10, y11 = get_y_elements(Y_defs[b])
        
        # Compute S0..S6
        # S0 = (x10+x11)(y00+y01)
        u_s0 = {**x10, **x11}
        for k, v in x11.items(): u_s0[k] = u_s0.get(k, 0) + v
        v_s0 = {**y00, **y01}
        for k, v in y01.items(): v_s0[k] = v_s0.get(k, 0) + v
        
        # S1 = (x10+x11)y00
        u_s1 = {**x10, **x11}
        for k, v in x11.items(): u_s1[k] = u_s1.get(k, 0) + v
        v_s1 = y00
        
        # S2 = x00(y10-y11)
        u_s2 = x00
        v_s2 = {**y10, **y11}
        for k, v in y11.items(): v_s2[k] = v_s2.get(k, 0) - v
        
        # S3 = x11y11
        u_s3 = x11
        v_s3 = y11
        
        # S4 = (x00+x11)(y00+y11)
        u_s4 = {**x00, **x11}
        for k, v in x11.items(): u_s4[k] = u_s4.get(k, 0) + v
        v_s4 = {**y00, **y11}
        for k, v in y11.items(): v_s4[k] = v_s4.get(k, 0) + v
        
        # S5 = (x01-x11)(y10+y11)
        u_s5 = {**x01, **x11}
        for k, v in x11.items(): u_s5[k] = u_s5.get(k, 0) - v
        v_s5 = {**y10, **y11}
        for k, v in y11.items(): v_s5[k] = v_s5.get(k, 0) + v
        
        # S6 = (x01-x00)(y00+y10)
        u_s6 = {**x01, **x00}
        for k, v in x00.items(): u_s6[k] = u_s6.get(k, 0) - v
        v_s6 = {**y00, **y10}
        for k, v in y10.items(): v_s6[k] = v_s6.get(k, 0) + v
        
        S_u = [u_s0, u_s1, u_s2, u_s3, u_s4, u_s5, u_s6]
        S_v = [v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6]
        
        # C elements
        # C00 = S4+S3-S1+S5
        # C01 = S0-S4+S6
        # C10 = S2+S3
        # C11 = S0-S2+S3-S1
        
        # C block indices (cols 0-3)
        # C00: k in 0,1; i in 0,1 -> 0, 1, 4, 5
        # C01: k in 0,1; i in 2,3 -> 2, 3, 6, 7
        # C10: k in 2,3; i in 0,1 -> 8, 9, 12, 13
        # C11: k in 2,3; i in 2,3 -> 10, 11, 14, 15
        
        c00 = [0, 1, 4, 5]
        c01 = [2, 3, 6, 7]
        c10 = [8, 9, 12, 13]
        c11 = [10, 11, 14, 15]
        
        # For each s in 0..6, we add the term S_u[s] * S_v[s] to the appropriate C elements.
        # We need to add U, V, W contributions.
        
        for s in range(7):
            # Add to U, V
            add_term(S_u[s], S_v[s], {}, col)
            
            # Determine W coefficients
            w_coeffs = {}
            
            # C00 gets S4, S3, -S1, S5
            if s == 4:
                for idx in c00: w_coeffs[idx] = w_coeffs.get(idx, 0) + 1
            if s == 3:
                for idx in c00: w_coeffs[idx] = w_coeffs.get(idx, 0) + 1
            if s == 1:
                for idx in c00: w_coeffs[idx] = w_coeffs.get(idx, 0) - 1
            if s == 5:
                for idx in c00: w_coeffs[idx] = w_coeffs.get(idx, 0) + 1
            
            # C01 gets S0, -S4, S6
            if s == 0:
                for idx in c01: w_coeffs[idx] = w_coeffs.get(idx, 0) + 1
            if s == 4:
                for idx in c01: w_coeffs[idx] = w_coeffs.get(idx, 0) - 1
            if s == 6:
                for idx in c01: w_coeffs[idx] = w_coeffs.get(idx, 0) + 1
            
            # C10 gets S2, S3
            if s == 2:
                for idx in c10: w_coeffs[idx] = w_coeffs.get(idx, 0) + 1
            if s == 3:
                for idx in c10: w_coeffs[idx] = w_coeffs.get(idx, 0) + 1
            
            # C11 gets S0, -S2, S3, -S1
            if s == 0:
                for idx in c11: w_coeffs[idx] = w_coeffs.get(idx, 0) + 1
            if s == 2:
                for idx in c11: w_coeffs[idx] = w_coeffs.get(idx, 0) - 1
            if s == 3:
                for idx in c11: w_coeffs[idx] = w_coeffs.get(idx, 0) + 1
            if s == 1:
                for idx in c11: w_coeffs[idx] = w_coeffs.get(idx, 0) - 1
            
            # Add W
            for idx, val in w_coeffs.items():
                if val != 0:
                    W[idx, col] += val
                    
            col += 1
            
    # --- Part 2: 4x4x1 Schoolbook (Rank 16) ---
    # Col 4 of B and C
    
    for j in range(4):
        for i in range(4):
            # A index: i*4 + j
            a_idx = i * 4 + j
            # B index: j*5 + 4
            b_idx = j * 5 + 4
            # C index: 4*4 + i (k=4, n=4)
            c_idx = 16 + i
            
            U[a_idx, col] = 1
            V[b_idx, col] = 1
            W[c_idx, col] = 1
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

