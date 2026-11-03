import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition for 4x4x4 (rank 49) and 4x4x2 (rank 28)
    # and concatenate them.
    
    # Helper to create Strassen decomposition for 2x2x2
    # Returns U, V, W of shape (4, 7), (4, 7), (4, 7)
    # Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    # Strassen formulas:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # We need to map these to the tensor format.
    # The tensor T[i,j, k,l, i',l'] = delta_{jk} delta_{ii'} delta_{ll'}
    # For Strassen, we are summing rank-1 tensors.
    # Each rank-1 tensor is defined by U_vec, V_vec, W_vec.
    # U_vec has non-zero entries at (i,j) positions.
    # V_vec has non-zero entries at (k,l) positions.
    # W_vec has non-zero entries at (i',l') positions.
    
    # Let's generate the 7 terms for 2x2.
    # i, j in {0,1}. k, l in {0,1}.
    # U index: i*2 + j
    # V index: k*2 + l
    # W index: l*2 + i  (Note: W[k*n+i] in problem statement? 
    # Problem says: W[k*n+i, r]. 
    # For 2x2, n=2. W index is k*2 + i.
    # Wait, the problem statement says: W[k*n+i, r].
    # My previous assumption was W[k*n+i]. 
    # Let's check the schoolbook code:
    # W[k * n + i][column] = 1
    # Yes, W index is k*n + i.
    # For 2x2, n=2. W index is 2*k + i.
    
    # Strassen Terms (U, V, W coefficients)
    # M1: U has (0,0)=1, (1,1)=1. V has (0,0)=1, (1,1)=1. W has (0,0)=1, (1,1)=1.
    # Wait, W index is 2*k + i.
    # For M1, the output is C11 += M1, C22 += M1.
    # C11 is i=0, l=0. W index 2*0 + 0 = 0.
    # C22 is i=1, l=1. W index 2*1 + 1 = 3.
    # So W has 1 at index 0 and 3.
    
    # Let's define the 7 terms explicitly for 2x2.
    # Term 1:
    # U: (0,0)=1, (1,1)=1 -> indices 0, 3
    # V: (0,0)=1, (1,1)=1 -> indices 0, 3
    # W: (0,0)=1, (1,1)=1 -> indices 0, 3
    # Term 2:
    # U: (1,0)=1, (1,1)=1 -> indices 2, 3
    # V: (0,0)=1 -> index 0
    # W: (0,1)=1 -> index 1 (2*0+1)
    # Term 3:
    # U: (0,0)=1 -> index 0
    # V: (0,1)=1, (1,1)=-1 -> indices 1, 3
    # W: (0,0)=1 -> index 0
    # Term 4:
    # U: (1,1)=1 -> index 3
    # V: (1,0)=1, (0,0)=-1 -> indices 2, 0
    # W: (1,0)=1 -> index 2 (2*1+0)
    # Term 5:
    # U: (0,0)=1, (0,1)=1 -> indices 0, 1
    # V: (1,1)=1 -> index 3
    # W: (1,1)=1 -> index 3
    # Term 6:
    # U: (1,0)=1, (0,0)=-1 -> indices 2, 0
    # V: (0,0)=1, (0,1)=1 -> indices 0, 1
    # W: (0,1)=1, (1,1)=1 -> indices 1, 3
    # Term 7:
    # U: (0,1)=1, (1,1)=-1 -> indices 1, 3
    # V: (1,0)=1, (1,1)=1 -> indices 2, 3
    # W: (1,0)=1, (1,1)=1 -> indices 2, 3

    # This is getting tedious to hardcode.
    # I will implement a generic recursive Strassen generator.
    
    def get_strassen_2x2():
        # Returns list of (U_dict, V_dict, W_dict)
        # U_dict: {(i,j): coeff}
        terms = []
        # T1
        terms.append({
            'U': {(0,0): 1, (1,1): 1},
            'V': {(0,0): 1, (1,1): 1},
            'W': {(0,0): 1, (1,1): 1}
        })
        # T2
        terms.append({
            'U': {(1,0): 1, (1,1): 1},
            'V': {(0,0): 1},
            'W': {(0,1): 1}
        })
        # T3
        terms.append({
            'U': {(0,0): 1},
            'V': {(0,1): 1, (1,1): -1},
            'W': {(0,0): 1}
        })
        # T4
        terms.append({
            'U': {(1,1): 1},
            'V': {(1,0): 1, (0,0): -1},
            'W': {(1,0): 1}
        })
        # T5
        terms.append({
            'U': {(0,0): 1, (0,1): 1},
            'V': {(1,1): 1},
            'W': {(1,1): 1}
        })
        # T6
        terms.append({
            'U': {(1,0): 1, (0,0): -1},
            'V': {(0,0): 1, (0,1): 1},
            'W': {(0,1): 1, (1,1): 1}
        })
        # T7
        terms.append({
            'U': {(0,1): 1, (1,1): -1},
            'V': {(1,0): 1, (1,1): 1},
            'W': {(1,0): 1, (1,1): 1}
        })
        return terms

    # Build 4x4x4 using recursive Strassen
    # A is 2x2 blocks of 2x2. B is 2x2 blocks of 2x2.
    # We use 7 Strassen terms at the block level.
    # Each block term is a 2x2x2 multiplication.
    
    # Block level Strassen terms (same as above, but indices are block indices)
    # Let block indices be (bi, bj) for A, (bk, bl) for B, (bi', bl') for C.
    # The 7 terms define which blocks of A and B are multiplied.
    # For each block multiplication, we use the 2x2 Strassen decomposition.
    
    # Total rank for 4x4x4 is 7 * 7 = 49.
    
    # Build 4x4x2
    # A is 2x2 blocks of 2x2. B is 2x1 blocks of 2x2.
    # Block level multiplication is 2x2 by 2x1.
    # We use the rank-4 algorithm for 2x2x1.
    # Terms for 2x2x1:
    # C1 = A11 B1 + A12 B2
    # C2 = A21 B1 + A22 B2
    # This is 4 terms:
    # 1. A11 * B1 -> C1
    # 2. A12 * B2 -> C1
    # 3. A21 * B1 -> C2
    # 4. A22 * B2 -> C2
    # Each is a 2x2x2 multiplication (rank 7).
    # Total rank 4 * 7 = 28.
    
    # Total rank 49 + 28 = 77.
    
    # Implementation details:
    # We need to construct U, V, W matrices of size (16, 77), (24, 77), (24, 77).
    # Note: V size is m*p = 4*6 = 24.
    # W size is p*n = 6*4 = 24.
    
    # For the 4x4x4 part:
    # V indices: k*4 + l? No, V is (m*p, rank). m=4, p=4.
    # V index is j*p + k? 
    # Problem says: V[j*p+k, r].
    # For 4x4, j in 0..3, k in 0..3.
    # In block terms, j = 2*bi + bj, k = 2*bk + bk_ (wait, k is the column of B).
    # B is 4x4. Rows 0..3, Cols 0..3.
    # Block (bk, bl) corresponds to rows 2*bk..2*bk+1, cols 2*bl..2*bl+1.
    # So j (row of B) = 2*bk + bk_inner.
    # k (col of B) = 2*bl + bl_inner.
    # V index = j*4 + k = (2*bk + bk_inner)*4 + (2*bl + bl_inner).
    
    # For the 4x4x2 part:
    # B is 4x2. Rows 0..3, Cols 0..1.
    # V index = j*2 + k? No, V is (m*p, rank). m=4, p=2.
    # V index = j*2 + k.
    # j = 2*bk + bk_inner.
    # k = bl_inner (0..1).
    
    # This requires careful index mapping.
    # I will write the code to generate these matrices.
    
    U = [[0]*77 for _ in range(16)]
    V = [[0]*77 for _ in range(24)]
    W = [[0]*77 for _ in range(24)]
    
    col = 0
    # Part 1: 4x4x4
    # 7 block terms
    block_terms_2x2 = get_strassen_2x2()
    for bt in block_terms_2x2:
        # bt defines which blocks of A and B are involved.
        # For each block multiplication, we iterate over the 2x2 Strassen terms.
        for st in block_terms_2x2:
            # st defines the inner 2x2 multiplication.
            # We need to combine the block indices and inner indices.
            # Block indices from bt:
            # U_block: {(bi, bj): coeff}
            # V_block: {(bk, bl): coeff}
            # W_block: {(bi', bl'): coeff}
            
            # Inner indices from st:
            # U_inner: {(i, j): coeff}
            # V_inner: {(k, l): coeff}
            # W_inner: {(i', l'): coeff}
            
            # Combined indices:
            # U: (bi, bj) -> (2*bi + i, 2*bj + j)
            # V: (bk, bl) -> (2*bk + k, 2*bl + l)
            # W: (bi', bl') -> (2*bl' + l', 2*bi' + i')? 
            # W index is k*n + i.
            # For 4x4, n=4.
            # W index = l*4 + i.
            # l is the column of C. i is the row of C.
            # In block terms, l = 2*bl' + l', i = 2*bi' + i'.
            # So W index = (2*bl' + l')*4 + (2*bi' + i').
            
            # We need to sum the coefficients for each rank-1 term.
            # Each combination of (bt, st) is one rank-1 term in the 4x4x4 decomposition.
            # There are 7*7=49 such terms.
            
            # Iterate over all non-zero entries in bt and st.
            # This is inefficient if done naively, but 49 terms is small.
            
            # Actually, each (bt, st) pair defines a single rank-1 tensor in the 4x4x4 space.
            # The U vector for this term has non-zeros at (2*bi+i, 2*bj+j) with coeff bt['U'][(bi,bj)] * st['U'][(i,j)].
            # Similarly for V and W.
            
            for (bi, bj), cu in bt['U'].items():
                for (i, j), su in st['U'].items():
                    u_idx = (2*bi + i)*4 + (2*bj + j)
                    U[u_idx][col] += cu * su
            
            for (bk, bl), cv in bt['V'].items():
                for (k, l), sv in st['V'].items():
                    v_idx = (2*bk + k)*4 + (2*bl + l)
                    V[v_idx][col] += cv * sv
            
            for (bi, bl), cw in bt['W'].items():
                # Note: bt['W'] keys are (bi', bl')?
                # In my get_strassen_2x2, W keys are (i, l) i.e. (row, col).
                # So bi is row block, bl is col block.
                for (i, l), sw in st['W'].items():
                    # W index = l*4 + i
                    w_idx = (2*bl + l)*4 + (2*bi + i)
                    W[w_idx][col] += cw * sw
            col += 1
            
    # Part 2: 4x4x2
    # 4 block terms (for 2x2x1)
    # Terms:
    # 1. A11 * B1 -> C1
    # 2. A12 * B2 -> C1
    # 3. A21 * B1 -> C2
    # 4. A22 * B2 -> C2
    
    # Block terms for 2x2x1:
    block_terms_2x1 = [
        {'U': {(0,0): 1}, 'V': {(0,0): 1}, 'W': {(0,0): 1}}, # A11 B1 -> C1
        {'U': {(0,1): 1}, 'V': {(1,0): 1}, 'W': {(0,0): 1}}, # A12 B2 -> C1 (B2 is block (1,0)?)
        # Wait, B is 2x1 blocks.
        # B1 is block (0,0). B2 is block (1,0).
        # C1 is block (0,0). C2 is block (1,0).
        # So:
        # 1. U(0,0) V(0,0) W(0,0)
        # 2. U(0,1) V(1,0) W(0,0)
        # 3. U(1,0) V(0,0) W(1,0)
        # 4. U(1,1) V(1,0) W(1,0)
    ]
    # Corrected block_terms_2x1:
    block_terms_2x1 = [
        {'U': {(0,0): 1}, 'V': {(0,0): 1}, 'W': {(0,0): 1}},
        {'U': {(0,1): 1}, 'V': {(1,0): 1}, 'W': {(0,0): 1}},
        {'U': {(1,0): 1}, 'V': {(0,0): 1}, 'W': {(1,0): 1}},
        {'U': {(1,1): 1}, 'V': {(1,0): 1}, 'W': {(1,0): 1}},
    ]
    
    for bt in block_terms_2x1:
        # Each block term involves a 2x2x2 multiplication.
        for st in block_terms_2x2:
            for (bi, bj), cu in bt['U'].items():
                for (i, j), su in st['U'].items():
                    u_idx = (2*bi + i)*4 + (2*bj + j)
                    U[u_idx][col] += cu * su
            
            for (bk, bl), cv in bt['V'].items():
                # V is 4x2. j in 0..3, k in 0..1.
                # Block (bk, bl) where bl is always 0?
                # B is 4x2. Rows 0..3, Cols 0..1.
                # Block (bk, 0) corresponds to rows 2*bk..2*bk+1, cols 0..1.
                # So bl is 0.
                # V index = j*2 + k.
                # j = 2*bk + k_inner.
                # k = l_inner (0..1).
                for (k, l), sv in st['V'].items():
                    # k is row of B inner, l is col of B inner.
                    # In st, V keys are (k, l) where k is row, l is col.
                    # For 2x2x2, V is 2x2.
                    # Here V block is 2x2 (rows 2*bk.., cols 0..1).
                    # So k_inner = k, l_inner = l.
                    v_idx = (2*bk + k)*2 + l
                    V[v_idx][col] += cv * sv
            
            for (bi, bl), cw in bt['W'].items():
                # W is 6x4? No, W is (p*n, rank) = (2*4, rank) = (8, rank)?
                # No, W is (p*n, rank). p=2, n=4. Size 8.
                # Wait, the problem says W is (p*n, rank).
                # For 4x4x2, p=2, n=4. W size 8.
                # But our total W size is 24 (for 4x4x6).
                # The 4x4x2 part uses the first 2 columns of B?
                # Yes, we split B into B1 (4x4) and B2 (4x2).
                # B2 corresponds to cols 4..5 of the original 4x6?
                # No, we split B into B1 (4x4) and B2 (4x2).
                # B1 uses cols 0..3. B2 uses cols 4..5?
                # Or B1 uses cols 0..3 and B2 uses cols 0..1 of a separate 4x2 matrix?
                # The problem is 4x4 by 4x6.
                # We can treat it as A * [B1, B2] where B1 is 4x4, B2 is 4x2.
                # So B2 columns are 4..5.
                # In the V matrix (size 24), the indices for B2 are j*6 + k where k in 4..5.
                # In my local 4x4x2 decomposition, V index is j*2 + k (k in 0..1).
                # So global V index = j*6 + (4 + k).
                
                # W index for 4x4x2:
                # W size 8. Index = l*4 + i (l in 0..1, i in 0..3).
                # Global W index?
                # W is (p*n, rank). p=6, n=4. Size 24.
                # W index = l*4 + i.
                # For B2, l is 4..5.
                # So global W index = l_global*4 + i.
                # l_global = 4 + l_local.
                
                for (i, l), sw in st['W'].items():
                    # st['W'] keys are (i, l) i.e. (row, col) of the 2x2 output block.
                    # For 4x4x2, output is 4x2.
                    # Block (bi, bl) where bl is 0?
                    # C is 4x2. Rows 0..3, Cols 0..1.
                    # Block (bi, 0) corresponds to rows 2*bi..2*bi+1, cols 0..1.
                    # So bl is 0.
                    # l_local = l.
                    # i_local = i.
                    l_global = 4 + l
                    i_global = 2*bi + i
                    w_idx = l_global*4 + i_global
                    W[w_idx][col] += cw * sw
            col += 1
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
