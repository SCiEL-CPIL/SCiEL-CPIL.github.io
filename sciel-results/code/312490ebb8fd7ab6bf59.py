import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Part 1: 4x4x4 using Strassen (Rank 49)
    # Part 2: 4x4x3 using Trivial (Rank 48)
    
    rank1 = 49
    rank2 = 48
    total_rank = rank1 + rank2
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Helper to get Strassen coefficients for 2x2x2
    # Returns list of 7 tuples: (coeff_A_00, coeff_A_01, coeff_A_10, coeff_A_11, 
    #                             coeff_B_00, coeff_B_01, coeff_B_10, coeff_B_11,
    #                             coeff_C_00, coeff_C_01, coeff_C_10, coeff_C_11)
    def get_strassen_coeffs_2x2():
        coeffs = []
        # M1 = (A00+A11)(B00+B11)
        coeffs.append((1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1)) # C00 += M1, C11 += M1
        # M2 = (A10+A11)B00
        coeffs.append((0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0)) # C10 += M2
        # M3 = A00(B01-B11)
        coeffs.append((1, 0, 0, 0, 0, 1, 0, -1, 0, 1, 0, 0)) # C01 += M3
        # M4 = A11(B10-B00)
        coeffs.append((0, 0, 0, 1, -1, 0, 1, 0, 1, 0, 1, 0)) # C00 += M4, C10 += M4
        # M5 = (A00+A01)B11
        coeffs.append((1, 1, 0, 0, 0, 0, 0, 1, -1, 1, 0, 0)) # C00 -= M5, C01 += M5
        # M6 = (A10-A00)(B00+B01)
        coeffs.append((-1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1)) # C11 += M6
        # M7 = (A01-A11)(B10+B11)
        coeffs.append((0, 1, 0, -1, 0, 0, 1, 1, 1, 0, 0, 0)) # C00 += M7
        
        # Correct C coefficients based on standard Strassen:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Re-evaluating the appended tuples above against standard formula.
        # The tuple format I used above was mixed. Let's define properly.
        pass

    # Let's just hardcode the 49 terms by iterating outer and inner steps.
    
    # Outer Strassen Steps for 4x4x4
    # A blocks: A00, A01, A10, A11 (2x2 each)
    # B blocks: B00, B01, B10, B11 (2x2 each)
    
    # Outer Coeffs:
    # M1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2: (A10+A11)B00       -> C10 += M2
    # M3: A00(B01-B11)       -> C01 += M3
    # M4: A11(B10-B00)       -> C00 += M4, C10 += M4
    # M5: (A00+A01)B11       -> C00 -= M5, C01 += M5
    # M6: (A10-A00)(B00+B01) -> C11 += M6
    # M7: (A01-A11)(B10+B11) -> C00 += M7
    
    outer_steps = [
        # (A_coeffs, B_coeffs, C_block_coeffs)
        # A_coeffs: dict of block_name -> coeff
        # B_coeffs: dict of block_name -> coeff
        # C_block_coeffs: dict of block_name -> coeff
        ({'A00': 1, 'A11': 1}, {'B00': 1, 'B11': 1}, {'C00': 1, 'C11': 1}),
        ({'A10': 1, 'A11': 1}, {'B00': 1}, {'C10': 1}),
        ({'A00': 1}, {'B01': 1, 'B11': -1}, {'C01': 1}),
        ({'A11': 1}, {'B10': 1, 'B00': -1}, {'C00': 1, 'C10': 1}),
        ({'A00': 1, 'A01': 1}, {'B11': 1}, {'C00': -1, 'C01': 1}),
        ({'A10': 1, 'A00': -1}, {'B00': 1, 'B01': 1}, {'C11': 1}),
        ({'A01': 1, 'A11': -1}, {'B10': 1, 'B11': 1}, {'C00': 1}),
    ]
    
    # Inner Strassen Steps for 2x2x2
    # A entries: A00, A01, A10, A11
    # B entries: B00, B01, B10, B11
    # C entries: C00, C01, C10, C11
    
    inner_steps = [
        ({'A00': 1, 'A11': 1}, {'B00': 1, 'B11': 1}, {'C00': 1, 'C11': 1}),
        ({'A10': 1, 'A11': 1}, {'B00': 1}, {'C10': 1}),
        ({'A00': 1}, {'B01': 1, 'B11': -1}, {'C01': 1}),
        ({'A11': 1}, {'B10': 1, 'B00': -1}, {'C00': 1, 'C10': 1}),
        ({'A00': 1, 'A01': 1}, {'B11': 1}, {'C00': -1, 'C01': 1}),
        ({'A10': 1, 'A00': -1}, {'B00': 1, 'B01': 1}, {'C11': 1}),
        ({'A01': 1, 'A11': -1}, {'B10': 1, 'B11': 1}, {'C00': 1}),
    ]
    
    col = 0
    for k, (A_outer, B_outer, C_outer) in enumerate(outer_steps):
        for l, (A_inner, B_inner, C_inner) in enumerate(inner_steps):
            # Construct U vector (size 16)
            u_vec = np.zeros(16)
            # Construct V vector (size 16)
            v_vec = np.zeros(16)
            # Construct W vector (size 16)
            w_vec = np.zeros(16)
            
            # Combine A coefficients
            # A_outer gives block coeffs. A_inner gives entry coeffs within block.
            # Total A coeff for entry (i,j) is sum over blocks b of (A_outer[b] * A_inner[b_entry])
            # Blocks: 00, 01, 10, 11
            # Entries in block b: 00, 01, 10, 11
            
            a_total = {}
            for block_name, coeff_o in A_outer.items():
                # Map block name to indices
                if block_name == 'A00':
                    b_i, b_j = 0, 0
                elif block_name == 'A01':
                    b_i, b_j = 0, 1
                elif block_name == 'A10':
                    b_i, b_j = 1, 0
                elif block_name == 'A11':
                    b_i, b_j = 1, 1
                else:
                    continue
                
                for entry_name, coeff_i in A_inner.items():
                    # entry_name is like 'A00' but relative to block
                    # We need to map entry_name to relative indices
                    # The keys in A_inner are 'A00', 'A01', 'A10', 'A11'
                    # These correspond to relative indices (0,0), (0,1), (1,0), (1,1)
                    e_i = int(entry_name[1])
                    e_j = int(entry_name[2])
                    
                    global_i = 2 * b_i + e_i
                    global_j = 2 * b_j + e_j
                    idx = global_i * 4 + global_j
                    
                    if idx not in a_total:
                        a_total[idx] = 0
                    a_total[idx] += coeff_o * coeff_i
            
            for idx, coeff in a_total.items():
                u_vec[idx] = coeff
                
            # Combine B coefficients
            b_total = {}
            for block_name, coeff_o in B_outer.items():
                if block_name == 'B00':
                    b_i, b_j = 0, 0
                elif block_name == 'B01':
                    b_i, b_j = 0, 1
                elif block_name == 'B10':
                    b_i, b_j = 1, 0
                elif block_name == 'B11':
                    b_i, b_j = 1, 1
                else:
                    continue
                
                for entry_name, coeff_i in B_inner.items():
                    e_i = int(entry_name[1])
                    e_j = int(entry_name[2])
                    
                    global_i = 2 * b_i + e_i
                    global_j = 2 * b_j + e_j
                    idx = global_i * 4 + global_j
                    
                    if idx not in b_total:
                        b_total[idx] = 0
                    b_total[idx] += coeff_o * coeff_i
            
            for idx, coeff in b_total.items():
                v_vec[idx] = coeff
                
            # Combine C coefficients
            # C_outer gives block coeffs. C_inner gives entry coeffs within block.
            c_total = {}
            for block_name, coeff_o in C_outer.items():
                if block_name == 'C00':
                    b_i, b_j = 0, 0
                elif block_name == 'C01':
                    b_i, b_j = 0, 1
                elif block_name == 'C10':
                    b_i, b_j = 1, 0
                elif block_name == 'C11':
                    b_i, b_j = 1, 1
                else:
                    continue
                
                for entry_name, coeff_i in C_inner.items():
                    e_i = int(entry_name[1])
                    e_j = int(entry_name[2])
                    
                    global_i = 2 * b_i + e_i
                    global_j = 2 * b_j + e_j
                    # W index: k*n + i. Here n=4.
                    # Wait, W[k*n+i, r].
                    # The output C is p x n.
                    # C is 4x4.
                    # Index in W is (global_j * 4 + global_i)?
                    # No, W[k*n+i]. k is column index of C, i is row index of C.
                    # In matrix multiplication C = AB, C[i,k] = sum_j A[i,j]B[j,k].
                    # So C row is i, col is k.
                    # W index is k * n + i.
                    idx_w = global_j * 4 + global_i
                    
                    if idx_w not in c_total:
                        c_total[idx_w] = 0
                    c_total[idx_w] += coeff_o * coeff_i
            
            for idx, coeff in c_total.items():
                w_vec[idx] = coeff
            
            U[:, col] = u_vec
            V[:, col] = v_vec
            W[:, col] = w_vec
            col += 1
            
    # Part 2: 4x4x3 Trivial
    # B is 4x3.
    # Indices for B in V: j (0..3), k (0..2).
    # Indices for C in W: k (0..2), i (0..3).
    # Indices for A in U: i (0..3), j (0..3).
    
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U[i * 4 + j, col] = 1
                V[j * 7 + k, col] = 1 # m=4, p=7. B is 4x7.
                # Wait, V shape is (m*p, rank) = (28, rank).
                # B is 4x7.
                # The trivial part uses the first 3 columns of B?
                # Yes, B_L is 4x4, B_R is 4x3.
                # B_R columns 0,1,2 of the remaining 3?
                # We split B (4x7) into B1 (4x4) and B2 (4x3).
                # B1 uses cols 0-3. B2 uses cols 4-6.
                # So for Part 2, B indices are j (0..3) and k (4..6).
                # V index: j * 7 + k.
                # W index: k * 4 + i.
                
                # Correction:
                # V[j * 7 + k, col] = 1
                # W[k * 4 + i, col] = 1
                
                V[j * 7 + k, col] = 1
                W[k * 4 + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
