import numpy as np

def construct():
    n, m, p = 4, 4, 5
    rank = 65
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Part 1: Rank 49 for 4x4x4
    # Define Strassen 2x2x2 decomposition
    # u, v, w are 4x7 matrices
    # Indices: u[i*2+j, r], v[j*2+k, r], w[k*2+i, r]
    
    # We will generate the 49 terms
    # We need the Strassen formulas for 2x2 blocks
    # M1: (A00+A11)(B00+B11) -> C00, C11
    # M2: (A10+A11)B00 -> C10, C11
    # M3: A00(B01-B11) -> C01, C11
    # M4: A11(B10-B00) -> C00, C10
    # M5: (A00+A01)B11 -> C00, C01
    # M6: (A10-A00)(B00+B01) -> C11
    # M7: (A01-A11)(B10+B11) -> C00, C11
    
    # For each Mk, we have a virtual 2x2 multiplication.
    # We use the standard 2x2x2 Strassen decomposition (rank 7) for the virtual multiplication.
    
    # Standard 2x2x2 Strassen (Rank 7):
    # S1 = (a00+a11)(b00+b11)
    # S2 = (a10+a11)b00
    # S3 = a00(b01-b11)
    # S4 = a11(b10-b00)
    # S5 = (a00+a01)b11
    # S6 = (a10-a00)(b00+b01)
    # S7 = (a01-a11)(b10+b11)
    
    # c00 = S1 + S4 - S5 + S7
    # c01 = S3 + S5
    # c10 = S2 + S4
    # c11 = S1 - S2 + S3 + S6
    
    # We define the 7 rank-1 terms for the virtual multiplication.
    # Each term is (u_vec, v_vec, w_vec) of length 4.
    
    def get_strassen_2x2_terms():
        terms = []
        # S1
        u1 = [1, 0, 0, 1] # a00, a11
        v1 = [1, 0, 0, 1] # b00, b11
        w1 = [1, 0, 0, 1] # c00, c11
        terms.append((u1, v1, w1))
        
        # S2
        u2 = [0, 0, 1, 1] # a10, a11
        v2 = [1, 0, 0, 0] # b00
        w2 = [0, 0, 1, 1] # c10, c11
        terms.append((u2, v2, w2))
        
        # S3
        u3 = [1, 0, 0, 0] # a00
        v3 = [0, 1, 0, -1] # b01, -b11
        w3 = [0, 1, 0, 1] # c01, c11
        terms.append((u3, v3, w3))
        
        # S4
        u4 = [0, 0, 0, 1] # a11
        v4 = [0, 0, 1, -1] # b10, -b00
        w4 = [1, 0, 1, 0] # c00, c10
        terms.append((u4, v4, w4))
        
        # S5
        u5 = [1, 1, 0, 0] # a00, a01
        v5 = [0, 0, 0, 1] # b11
        w5 = [1, 1, 0, 0] # c00, c01
        terms.append((u5, v5, w5))
        
        # S6
        u6 = [0, 0, 1, -1] # a10, -a00
        v6 = [1, 1, 0, 0] # b00, b01
        w6 = [0, 0, 0, 1] # c11
        terms.append((u6, v6, w6))
        
        # S7
        u7 = [0, 1, 0, -1] # a01, -a11
        v7 = [0, 0, 1, 1] # b10, b11
        w7 = [1, 0, 0, 1] # c00, c11
        terms.append((u7, v7, w7))
        
        return terms

    strassen_2x2 = get_strassen_2x2_terms()
    
    # Now map the 7 block multiplications to the virtual 2x2 terms
    # Each block multiplication Mk has:
    # A_comb: dict mapping virtual A index (0..3) to list of (real_A_index, coeff)
    # B_comb: dict mapping virtual B index (0..3) to list of (real_B_index, coeff)
    # C_comb: dict mapping virtual C index (0..3) to list of (real_C_index, coeff)
    
    # Real A indices: 0..15 (i*4+j)
    # Real B indices: 0..19 (j*5+k)
    # Real C indices: 0..19 (k*4+i)
    
    # Block A00: indices 0,1,4,5 (i=0,1; j=0,1)
    # Block A01: indices 2,3,6,7 (i=0,1; j=2,3)
    # Block A10: indices 8,9,12,13 (i=2,3; j=0,1)
    # Block A11: indices 10,11,14,15 (i=2,3; j=2,3)
    
    # Block B00: indices 0,1,5,6 (j=0,1; k=0,1)
    # Block B01: indices 2,3,7,8 (j=0,1; k=2,3)
    # Block B10: indices 10,11,15,16 (j=2,3; k=0,1)
    # Block B11: indices 12,13,17,18 (j=2,3; k=2,3)
    
    # Block C00: indices 0,1,4,5 (k=0,1; i=0,1)
    # Block C01: indices 2,3,6,7 (k=2,3; i=0,1)
    # Block C10: indices 8,9,12,13 (k=0,1; i=2,3)
    # Block C11: indices 10,11,14,15 (k=2,3; i=2,3)
    
    # Define the 7 block multiplications
    # Mk: (A_blocks, B_blocks, C_blocks)
    # A_blocks: list of (block_name, coeff)
    # B_blocks: list of (block_name, coeff)
    # C_blocks: list of (block_name, coeff)
    
    # M1: (A00+A11)(B00+B11) -> C00, C11
    # M2: (A10+A11)B00 -> C10, C11
    # M3: A00(B01-B11) -> C01, C11
    # M4: A11(B10-B00) -> C00, C10
    # M5: (A00+A01)B11 -> C00, C01
    # M6: (A10-A00)(B00+B01) -> C11
    # M7: (A01-A11)(B10+B11) -> C00, C11
    
    block_mappings = [
        { # M1
            'A': [('A00', 1), ('A11', 1)],
            'B': [('B00', 1), ('B11', 1)],
            'C': [('C00', 1), ('C11', 1)]
        },
        { # M2
            'A': [('A10', 1), ('A11', 1)],
            'B': [('B00', 1)],
            'C': [('C10', 1), ('C11', 1)]
        },
        { # M3
            'A': [('A00', 1)],
            'B': [('B01', 1), ('B11', -1)],
            'C': [('C01', 1), ('C11', 1)]
        },
        { # M4
            'A': [('A11', 1)],
            'B': [('B10', 1), ('B00', -1)],
            'C': [('C00', 1), ('C10', 1)]
        },
        { # M5
            'A': [('A00', 1), ('A01', 1)],
            'B': [('B11', 1)],
            'C': [('C00', 1), ('C01', 1)]
        },
        { # M6
            'A': [('A10', 1), ('A00', -1)],
            'B': [('B00', 1), ('B01', 1)],
            'C': [('C11', 1)]
        },
        { # M7
            'A': [('A01', 1), ('A11', -1)],
            'B': [('B10', 1), ('B11', 1)],
            'C': [('C00', 1), ('C11', 1)]
        }
    ]
    
    # Helper to get real indices for a block
    def get_block_indices(block_name, type):
        # type: 'A', 'B', 'C'
        if type == 'A':
            if block_name == 'A00': return [0, 1, 4, 5]
            if block_name == 'A01': return [2, 3, 6, 7]
            if block_name == 'A10': return [8, 9, 12, 13]
            if block_name == 'A11': return [10, 11, 14, 15]
        elif type == 'B':
            if block_name == 'B00': return [0, 1, 5, 6]
            if block_name == 'B01': return [2, 3, 7, 8]
            if block_name == 'B10': return [10, 11, 15, 16]
            if block_name == 'B11': return [12, 13, 17, 18]
        elif type == 'C':
            if block_name == 'C00': return [0, 1, 4, 5]
            if block_name == 'C01': return [2, 3, 6, 7]
            if block_name == 'C10': return [8, 9, 12, 13]
            if block_name == 'C11': return [10, 11, 14, 15]
        return []
    
    current_rank = 0
    for mapping in block_mappings:
        # For each of the 7 virtual Strassen terms
        for u_v, v_v, w_v in strassen_2x2:
            # Construct real U, V, W vectors
            u_real = [0] * 16
            v_real = [0] * 20
            w_real = [0] * 20
            
            # Expand U
            for vi_idx, val in enumerate(u_v):
                if val == 0: continue
                # vi_idx is virtual A index 0..3
                # We need to map this to the real A indices involved in this block multiplication
                # The mapping is defined by mapping['A']
                # But wait, the virtual A index corresponds to a specific position in the 2x2 virtual matrix.
                # The block mapping tells us which real blocks form the virtual blocks.
                # However, the virtual 2x2 matrix is formed by concatenating the blocks?
                # No, the virtual A is a 2x2 matrix where each entry is a 2x2 block?
                # No, in Strassen for 4x4, the "entries" of the 2x2 meta-matrix are 2x2 blocks.
                # So $A_{virt}[0,0]$ is the block $A_{00}$.
                # $A_{virt}[0,1]$ is the block $A_{01}$.
                # $A_{virt}[1,0]$ is the block $A_{10}$.
                # $A_{virt}[1,1]$ is the block $A_{11}$.
                
                # So, if $u_v[0] \neq 0$, it affects the block $A_{00}$.
                # The block $A_{00}$ is composed of real indices [0,1,4,5].
                # The coefficient for each of these real indices is $val \times (\text{coeff from mapping})$.
                
                # Wait, the mapping is a list of (block_name, coeff).
                # We need to find which block corresponds to the virtual index.
                # Virtual index 0 -> A00
                # Virtual index 1 -> A01
                # Virtual index 2 -> A10
                # Virtual index 3 -> A11
                
                block_names = ['A00', 'A01', 'A10', 'A11']
                b_name = block_names[vi_idx]
                
                # Find coeff for this block in mapping['A']
                coeff = 0
                for b, c in mapping['A']:
                    if b == b_name:
                        coeff = c
                        break
                
                if coeff == 0: continue
                
                real_indices = get_block_indices(b_name, 'A')
                for r_idx in real_indices:
                    u_real[r_idx] += val * coeff
            
            # Expand V
            for vi_idx, val in enumerate(v_v):
                if val == 0: continue
                block_names = ['B00', 'B01', 'B10', 'B11']
                b_name = block_names[vi_idx]
                
                coeff = 0
                for b, c in mapping['B']:
                    if b == b_name:
                        coeff = c
                        break
                
                if coeff == 0: continue
                
                real_indices = get_block_indices(b_name, 'B')
                for r_idx in real_indices:
                    v_real[r_idx] += val * coeff
            
            # Expand W
            for vi_idx, val in enumerate(w_v):
                if val == 0: continue
                block_names = ['C00', 'C01', 'C10', 'C11']
                b_name = block_names[vi_idx]
                
                coeff = 0
                for b, c in mapping['C']:
                    if b == b_name:
                        coeff = c
                        break
                
                if coeff == 0: continue
                
                real_indices = get_block_indices(b_name, 'C')
                for r_idx in real_indices:
                    w_real[r_idx] += val * coeff
            
            # Add to U, V, W
            U[:, current_rank] = u_real
            V[:, current_rank] = v_real
            W[:, current_rank] = w_real
            current_rank += 1
            
    # Part 2: Rank 16 for 5th column
    # C[i,4] = sum_j A[i,j] B[j,4]
    for i in range(4):
        for j in range(4):
            U[i*4+j, current_rank] = 1
            V[j*5+4, current_rank] = 1
            W[4*4+i, current_rank] = 1
            current_rank += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
