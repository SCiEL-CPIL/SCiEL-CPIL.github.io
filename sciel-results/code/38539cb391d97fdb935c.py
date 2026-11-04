import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # We will build U, V, W as lists of lists
    U = []
    V = []
    W = []
    
    # Helper to get index in flattened 4x4 matrix
    # A[i, j] -> i * 4 + j
    # Blocks:
    # A11: (0,0)-(1,1)
    # A12: (0,2)-(1,3)
    # A21: (2,0)-(3,1)
    # A22: (2,2)-(3,3)
    
    # Define block indices
    # b11: 0,1,4,5
    # b12: 2,3,6,7
    # b21: 8,9,12,13
    # b22: 10,11,14,15
    
    # Strassen outer coefficients
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B12-B22)
    # P4 = A22(B21-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    
    # Coefficients for A blocks in Xk
    # Xk is a 2x2 matrix of blocks
    # We define Xk as a dict of block_name -> coeff
    # Then expand to scalar indices
    
    block_to_indices = {
        '11': [0, 1, 4, 5],
        '12': [2, 3, 6, 7],
        '21': [8, 9, 12, 13],
        '22': [10, 11, 14, 15]
    }
    
    # Outer Strassen definitions
    # k: (X_blocks, Y_blocks)
    # X_blocks: dict {block: coeff}
    outer_strassen = [
        ({'11': 1, '22': 1}, {'11': 1, '22': 1}), # P1
        ({'21': 1, '22': 1}, {'11': 1}),          # P2
        ({'11': 1}, {'12': 1, '22': -1}),         # P3
        ({'22': 1}, {'21': 1, '11': -1}),         # P4
        ({'11': 1, '12': 1}, {'22': 1}),          # P5
        ({'21': 1, '11': -1}, {'11': 1, '12': 1}),# P6
        ({'12': 1, '22': -1}, {'21': 1, '22': 1}) # P7
    ]
    
    # Inner Strassen definitions for 2x2 matrices
    # Each term s is defined by (coeffs for X elements, coeffs for Y elements)
    # X = [[x00, x01], [x10, x11]]
    # Y = [[y00, y01], [y10, y11]]
    # S1 = (x00+x11)(y00+y11)
    # S2 = (x10+x11)y00
    # S3 = x00(y01-y11)
    # S4 = x11(y10-y00)
    # S5 = (x00+x01)y11
    # S6 = (x10-x00)(y00+y01)
    # S7 = (x01-x11)(y10+y11)
    
    # Representing X elements as (row, col)
    # x00: (0,0), x01: (0,1), x10: (1,0), x11: (1,1)
    
    inner_strassen = [
        ([(0,0), 1], [(1,1), 1], [(0,0), 1], [(1,1), 1]), # S1: (x00+x11)(y00+y11)
        ([(1,0), 1], [(1,1), 1], [(0,0), 1], None),        # S2: (x10+x11)y00
        ([(0,0), 1], None, [(0,1), 1], [(1,1), -1]),       # S3: x00(y01-y11)
        ([(1,1), 1], None, [(1,0), 1], [(0,0), -1]),       # S4: x11(y10-y00)
        ([(0,0), 1], [(0,1), 1], [(1,1), 1], None),        # S5: (x00+x01)y11
        ([(1,0), 1], [(0,0), -1], [(0,0), 1], [(0,1), 1]), # S6: (x10-x00)(y00+y01)
        ([(0,1), 1], [(1,1), -1], [(1,0), 1], [(1,1), 1])  # S7: (x01-x11)(y10+y11)
    ]
    
    # Coefficients of Sk in Pk elements
    # Pk = [[p00, p01], [p10, p11]]
    # p00 = S1 + S4 - S5 + S7
    # p01 = S3 + S5
    # p10 = S2 + S4
    # p11 = S1 - S2 + S3 + S6
    
    p_coeffs = [
        [1, 0, 0, 1, -1, 0, 1], # p00
        [0, 0, 1, 0, 1, 0, 0],  # p01
        [0, 1, 0, 1, 0, 0, 0],  # p10
        [1, -1, 1, 0, 0, 1, 0]  # p11
    ]
    
    # Coefficients of Pk in C blocks
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # c_coeffs[k][block][p, q]
    # k is 0..6 (P1..P7)
    # block is '11', '12', '21', '22'
    # p, q is 0..1
    
    c_coeffs = {}
    for k in range(7):
        c_coeffs[k] = {}
        for b in ['11', '12', '21', '22']:
            c_coeffs[k][b] = [[0.0 for _ in range(2)] for _ in range(2)]
            
    # P1 (k=0)
    c_coeffs[0]['11'] = [[1, 0], [0, 1]]
    c_coeffs[0]['22'] = [[1, 0], [0, 1]]
    # P2 (k=1)
    c_coeffs[1]['21'] = [[1, 0], [0, 1]]
    c_coeffs[1]['22'] = [[-1, 0], [0, -1]]
    # P3 (k=2)
    c_coeffs[2]['12'] = [[1, 0], [0, 1]]
    c_coeffs[2]['22'] = [[1, 0], [0, 1]]
    # P4 (k=3)
    c_coeffs[3]['11'] = [[1, 0], [0, 1]]
    c_coeffs[3]['21'] = [[1, 0], [0, 1]]
    # P5 (k=4)
    c_coeffs[4]['11'] = [[-1, 0], [0, -1]]
    c_coeffs[4]['12'] = [[1, 0], [0, 1]]
    # P6 (k=5)
    c_coeffs[5]['22'] = [[1, 0], [0, 1]]
    # P7 (k=6)
    c_coeffs[6]['11'] = [[1, 0], [0, 1]]
    
    rank = 0
    for k in range(7):
        X_blocks, Y_blocks = outer_strassen[k]
        for s in range(7):
            # Get U vector
            u_vec = [0.0] * 16
            # Get V vector
            v_vec = [0.0] * 16
            
            # Process X part
            x_elements, y_elements = inner_strassen[s][:2] # This is wrong, inner_strassen structure needs adjustment
            
            # Let's re-structure inner_strassen to be clearer
            # s: (x_terms, y_terms)
            # x_terms: list of (pos, coeff)
            
            # Re-defining inner_strassen properly in code below
            pass
            
    # Corrected inner_strassen definition
    inner_strassen = [
        ([(0,0), 1], [(1,1), 1], [(0,0), 1], [(1,1), 1]), 
        ([(1,0), 1], [(1,1), 1], [(0,0), 1], None),        
        ([(0,0), 1], None, [(0,1), 1], [(1,1), -1]),       
        ([(1,1), 1], None, [(1,0), 1], [(0,0), -1]),       
        ([(0,0), 1], [(0,1), 1], [(1,1), 1], None),        
        ([(1,0), 1], [(0,0), -1], [(0,0), 1], [(0,1), 1]), 
        ([(0,1), 1], [(1,1), -1], [(1,0), 1], [(1,1), 1])  
    ]
    # Format: (x_terms, y_terms) where x_terms is list of (pos, coeff)
    # My previous definition was mixed. Let's make it (x_terms, y_terms)
    
    inner_strassen = [
        ([(0,0), 1], [(1,1), 1], [(0,0), 1], [(1,1), 1]), 
        # S1: x: [(0,0),1], [(1,1),1]; y: [(0,0),1], [(1,1),1]
        ([(1,0), 1], [(1,1), 1], [(0,0), 1], None),
        # S2: x: [(1,0),1], [(1,1),1]; y: [(0,0),1]
        ([(0,0), 1], None, [(0,1), 1], [(1,1), -1]),
        # S3: x: [(0,0),1]; y: [(0,1),1], [(1,1),-1]
        ([(1,1), 1], None, [(1,0), 1], [(0,0), -1]),
        # S4: x: [(1,1),1]; y: [(1,0),1], [(0,0),-1]
        ([(0,0), 1], [(0,1), 1], [(1,1), 1], None),
        # S5: x: [(0,0),1], [(0,1),1]; y: [(1,1),1]
        ([(1,0), 1], [(0,0), -1], [(0,0), 1], [(0,1), 1]),
        # S6: x: [(1,0),1], [(0,0),-1]; y: [(0,0),1], [(0,1),1]
        ([(0,1), 1], [(1,1), -1], [(1,0), 1], [(1,1), 1])
        # S7: x: [(0,1),1], [(1,1),-1]; y: [(1,0),1], [(1,1),1]
    ]
    
    # This format is still messy. Let's use dicts.
    inner_strassen = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1})
    ]
    
    for k in range(7):
        X_blocks, Y_blocks = outer_strassen[k]
        for s in range(7):
            x_terms, y_terms = inner_strassen[s]
            
            u_vec = [0.0] * 16
            v_vec = [0.0] * 16
            w_vec = [0.0] * 16
            
            # Calculate u_vec
            for pos, coeff_x in x_terms.items():
                r, c = pos
                # Get the block at (r, c) from X_blocks
                # X_blocks is a dict of block_name -> coeff
                # pos (0,0) -> block '11'
                # pos (0,1) -> block '12'
                # pos (1,0) -> block '21'
                # pos (1,1) -> block '22'
                block_name = f"{r+1}{c+1}"
                coeff_block = X_blocks.get(block_name, 0)
                if coeff_block != 0:
                    total_coeff = coeff_x * coeff_block
                    for idx in block_to_indices[block_name]:
                        u_vec[idx] += total_coeff
                        
            # Calculate v_vec
            for pos, coeff_y in y_terms.items():
                r, c = pos
                block_name = f"{r+1}{c+1}"
                coeff_block = Y_blocks.get(block_name, 0)
                if coeff_block != 0:
                    total_coeff = coeff_y * coeff_block
                    for idx in block_to_indices[block_name]:
                        v_vec[idx] += total_coeff
                        
            # Calculate w_vec
            # Sum over p, q of p_coeffs[s][p][q] * c_coeffs[k][block][p][q]
            # Wait, p_coeffs[s] is the coefficient of S_s in P_k[p,q]
            # c_coeffs[k][block] is the coefficient of P_k in C_block
            # So contribution to C_block[p,q] is sum_s p_coeffs[s][p][q] * c_coeffs[k][block][p][q] ?
            # No, for a fixed s, the contribution to C_block[p,q] is:
            # coeff = p_coeffs[s][p][q] * c_coeffs[k][block][p][q]
            # And we sum this for all p, q?
            # No, P_k[p,q] is a scalar. C_block is a matrix.
            # C_block = sum_k c_coeffs[k][block] * P_k
            # P_k[p,q] = sum_s p_coeffs[s][p][q] * S_s
            # So C_block[p,q] = sum_k sum_s c_coeffs[k][block][p][q] * p_coeffs[s][p][q] * S_s
            # For a fixed k, s, the coefficient of S_s in C_block[p,q] is:
            # c_coeffs[k][block][p][q] * p_coeffs[s][p][q]
            
            for block in ['11', '12', '21', '22']:
                for p in range(2):
                    for q in range(2):
                        coeff_c = c_coeffs[k][block][p][q]
                        coeff_p = p_coeffs[s][p][q]
                        if coeff_c != 0 and coeff_p != 0:
                            total_coeff = coeff_c * coeff_p
                            # Add to w_vec at the index corresponding to C_block[p,q]
                            # C_block '11' is indices 0,1,4,5
                            # p=0,q=0 -> idx 0
                            # p=0,q=1 -> idx 1
                            # p=1,q=0 -> idx 4
                            # p=1,q=1 -> idx 5
                            # General mapping:
                            # block '11' base 0, '12' base 2, '21' base 8, '22' base 10
                            # offset = p*4 + q? No.
                            # A[i,j] -> i*4+j
                            # Block '11' (0,0) to (1,1)
                            # p=0,q=0 -> (0,0) -> 0
                            # p=0,q=1 -> (0,1) -> 1
                            # p=1,q=0 -> (1,0) -> 4
                            # p=1,q=1 -> (1,1) -> 5
                            # Block '12' (0,2) to (1,3)
                            # p=0,q=0 -> (0,2) -> 2
                            # p=0,q=1 -> (0,3) -> 3
                            # p=1,q=0 -> (1,2) -> 6
                            # p=1,q=1 -> (1,3) -> 7
                            
                            base_idx = block_to_indices[block][0]
                            # The indices in block_to_indices are sorted.
                            # idx = base_idx + p*? + q*?
                            # For '11': 0,1,4,5. p=0 -> 0,1. p=1 -> 4,5.
                            # Offset for p is 4. Offset for q is 1.
                            # idx = base_idx + p*4 + q
                            idx = base_idx + p*4 + q
                            w_vec[idx] += total_coeff
            
            U.append(u_vec)
            V.append(v_vec)
            W.append(w_vec)
            rank += 1
            
    return {"U": U, "V": V, "W": W}
