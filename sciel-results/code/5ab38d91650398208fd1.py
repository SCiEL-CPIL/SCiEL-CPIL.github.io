import numpy as np
import time

def _strassen_2x2_2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7.
    U: (4, 7)
    V: (4, 7)
    W: (4, 7)
    Indices:
    A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # m1 = (a+d)(e+h)
    # a=0, d=3, e=0, h=3
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # C00 += 1, C11 += 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # m2 = (c+d)e
    # c=2, d=3, e=0
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # C10 += 1
    W[2, 1] = 1
    
    # m3 = a(g-h)
    # a=0, g=2, h=3
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    # C01 += 1
    W[1, 2] = 1
    
    # m4 = d(f-e)
    # d=3, f=1, e=0
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    # C00 += 1, C10 += 1
    W[0, 3] = 1; W[2, 3] = 1
    
    # m5 = (a+b)h
    # a=0, b=1, h=3
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # C00 -= 1, C01 += 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # m6 = (c-a)(e+f)
    # c=2, a=0, e=0, f=1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # C11 += 1
    W[3, 5] = 1
    
    # m7 = (b-d)(g+h)
    # b=1, d=3, g=2, h=3
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    # C00 += 1
    W[0, 6] = 1
    
    return U, V, W

def _strassen_4x4_4x4():
    """
    Returns U, V, W for 4x4x4 matrix multiplication with rank 49.
    Uses 7 Strassen products, each rank 7.
    """
    U_2, V_2, W_2 = _strassen_2x2_2x2()
    
    # Map local 2x2 block indices to global 4x4 indices
    # A blocks:
    # A00: rows 0-1, cols 0-1 -> global 0-3
    # A01: rows 0-1, cols 2-3 -> global 4-7
    # A10: rows 2-3, cols 0-1 -> global 8-11
    # A11: rows 2-3, cols 2-3 -> global 12-15
    
    # B blocks:
    # B00: rows 0-1, cols 0-1 -> global 0-3 (cols 0,1)
    # B01: rows 0-1, cols 2-3 -> global 4-7 (cols 2,3)
    # B10: rows 2-3, cols 0-1 -> global 14-17 (cols 0,1, rows 2,3 -> 2*4+0=8? No. B is 4x4. Index j*4+k.
    # B is 4x4. j is row, k is col.
    # B00: j=0,1; k=0,1. Indices: 0,1, 4,5.
    # B01: j=0,1; k=2,3. Indices: 2,3, 6,7.
    # B10: j=2,3; k=0,1. Indices: 8,9, 12,13.
    # B11: j=2,3; k=2,3. Indices: 10,11, 14,15.
    
    # C blocks:
    # C00: rows 0-1, cols 0-1. C index k*4+i.
    # C00: k=0,1; i=0,1. Indices: 0,1, 4,5.
    # C01: k=2,3; i=0,1. Indices: 2,3, 6,7.
    # C10: k=0,1; i=2,3. Indices: 8,9, 12,13.
    # C11: k=2,3; i=2,3. Indices: 10,11, 14,15.
    
    # Strassen Products M1..M7
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We need to construct U, V, W for each product.
    # Each product is a 2x2x2 multiplication.
    # The inputs are linear combinations of the blocks.
    # So for each term in the 2x2x2 decomposition, we need to map the coefficients to the global U, V.
    
    U_4 = np.zeros((16, 49))
    V_4 = np.zeros((16, 49))
    W_4 = np.zeros((16, 49))
    
    # Helper to add a scaled 2x2x2 decomposition
    def add_product(u_scale, v_scale, w_scale, offset):
        # u_scale: dict {local_idx: scale}
        # v_scale: dict {local_idx: scale}
        # w_scale: dict {local_idx: scale}
        for r in range(7):
            for i in range(4):
                if U_2[i, r] != 0:
                    U_4[u_scale[i], offset + r] += U_2[i, r] * u_scale[i] # Wait, u_scale applies to the whole block?
                    # No, U_2[i,r] is the coefficient of A_local[i] in the linear form.
                    # The linear form is sum A_local[i] * U_2[i,r].
                    # A_local[i] is a block entry.
                    # The global U index is the global position of A_local[i].
                    # The value is U_2[i,r] * (scale of the block in the linear combination).
                    pass
            
    # This is getting complicated to generalize. Let's just hardcode the 7 products.
    # Product 1: (A00+A11)(B00+B11)
    # A part: A00 + A11.
    # B part: B00 + B11.
    # C part: C00 += 1, C11 += 1 (from m1).
    
    # Let's define the mapping for A blocks to global indices.
    # A00: [0,1,4,5]
    # A01: [2,3,6,7]
    # A10: [8,9,12,13]
    # A11: [10,11,14,15]
    
    # B00: [0,1,4,5]
    # B01: [2,3,6,7]
    # B10: [8,9,12,13]
    # B11: [10,11,14,15]
    
    # C00: [0,1,4,5]
    # C01: [2,3,6,7]
    # C10: [8,9,12,13]
    # C11: [10,11,14,15]
    
    # For each Strassen product, we define the linear combination of A blocks and B blocks.
    # Then we use the 2x2x2 decomposition.
    
    products = [
        # (A_blocks, B_blocks, C_blocks, C_coeffs)
        # A_blocks: list of (block_name, coeff)
        # B_blocks: list of (block_name, coeff)
        # C_blocks: list of (block_name, coeff)
        # But C_coeffs depend on the specific m_k term.
        # So we need to iterate over the 7 terms of Strassen.
    ]
    
    # Strassen terms for 2x2x2:
    # m1: A=(0,1), (3,1); B=(0,1), (3,1); C=(0,1), (3,1) -> C00, C11
    # m2: A=(2,1), (3,1); B=(0,1); C=(2,1) -> C10
    # m3: A=(0,1); B=(2,1), (3,-1); C=(1,1) -> C01
    # m4: A=(3,1); B=(1,1), (0,-1); C=(0,1), (2,1) -> C00, C10
    # m5: A=(0,1), (1,1); B=(3,1); C=(0,-1), (1,1) -> C00, C01
    # m6: A=(2,1), (0,-1); B=(0,1), (1,1); C=(3,1) -> C11
    # m7: A=(1,1), (3,-1); B=(2,1), (3,1); C=(0,1) -> C00
    
    # We need to map these local indices to global block indices.
    # Local A indices: 0:A00, 1:A01, 2:A10, 3:A11.
    # Local B indices: 0:B00, 1:B01, 2:B10, 3:B11.
    # Local C indices: 0:C00, 1:C01, 2:C10, 3:C11.
    
    # For each Strassen product M_p, the A input is a linear combo of blocks.
    # e.g. M1 A input: A00 + A11.
    # This means for local index 0, coeff is 1. For local index 3, coeff is 1.
    # For local index 1, 2, coeff is 0.
    
    # We will loop over the 7 Strassen products.
    # For each product, we have a specific A-combo and B-combo.
    # Then we loop over the 7 terms of the 2x2x2 decomposition.
    # For each term, we have specific local A, B, C indices and coefficients.
    # We combine the A-combo coeff and the term coeff to get the global U coeff.
    
    strassen_products = [
        # (A_coeffs, B_coeffs)
        ({0: 1, 3: 1}, {0: 1, 3: 1}), # M1
        ({2: 1, 3: 1}, {0: 1}),       # M2
        ({0: 1}, {2: 1, 3: -1}),      # M3
        ({3: 1}, {1: 1, 0: -1}),      # M4
        ({0: 1, 1: 1}, {3: 1}),       # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}),# M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}) # M7
    ]
    
    # C coeffs for each m_k term (from 2x2x2 decomposition)
    # m1: C00=1, C11=1
    # m2: C10=1
    # m3: C01=1
    # m4: C00=1, C10=1
    # m5: C00=-1, C01=1
    # m6: C11=1
    # m7: C00=1
    
    c_coeffs = [
        {0: 1, 3: 1},
        {2: 1},
        {1: 1},
        {0: 1, 2: 1},
        {0: -1, 1: 1},
        {3: 1},
        {0: 1}
    ]
    
    # Block to Global Index Mapping
    # A Blocks
    A_map = {
        0: [0, 1, 4, 5],   # A00
        1: [2, 3, 6, 7],   # A01
        2: [8, 9, 12, 13], # A10
        3: [10, 11, 14, 15] # A11
    }
    # B Blocks
    B_map = {
        0: [0, 1, 4, 5],   # B00
        1: [2, 3, 6, 7],   # B01
        2: [8, 9, 12, 13], # B10
        3: [10, 11, 14, 15] # B11
    }
    # C Blocks
    C_map = {
        0: [0, 1, 4, 5],   # C00
        1: [2, 3, 6, 7],   # C01
        2: [8, 9, 12, 13], # C10
        3: [10, 11, 14, 15] # C11
    }
    
    offset = 0
    for p_idx in range(7):
        a_combo, b_combo = strassen_products[p_idx]
        for t_idx in range(7):
            # Get local indices and coeffs for this term
            u_local = U_2[:, t_idx]
            v_local = V_2[:, t_idx]
            w_local = W_2[:, t_idx]
            
            # Construct global U
            for i in range(4):
                if u_local[i] != 0:
                    # i is local block index
                    if i in a_combo:
                        coeff = u_local[i] * a_combo[i]
                        # Map to global indices
                        for g_idx in A_map[i]:
                            U_4[g_idx, offset + t_idx] += coeff
            
            # Construct global V
            for i in range(4):
                if v_local[i] != 0:
                    if i in b_combo:
                        coeff = v_local[i] * b_combo[i]
                        for g_idx in B_map[i]:
                            V_4[g_idx, offset + t_idx] += coeff
            
            # Construct global W
            # c_coeffs[t_idx] gives C block index and coeff
            for c_block, c_coeff in c_coeffs[t_idx].items():
                # w_local[i] is the coeff for local C block i?
                # No, W_2[:, t_idx] corresponds to local C indices 0..3.
                # So we need to check which local C indices are non-zero in w_local.
                for i in range(4):
                    if w_local[i] != 0 and i == c_block:
                         # Wait, c_coeffs is fixed for the term.
                         # w_local[i] is the coefficient in the 2x2x2 decomposition.
                         # The total coefficient is w_local[i] * c_coeff?
                         # No, the 2x2x2 decomposition already includes the signs.
                         # c_coeffs is redundant if we use W_2 directly?
                         # Yes, W_2[i, t_idx] is the coefficient for C_local[i].
                         # So we just map W_2 to global C.
                         pass
            
            # Correct W construction:
            for i in range(4):
                if w_local[i] != 0:
                    # i is local C block index
                    coeff = w_local[i]
                    for g_idx in C_map[i]:
                        W_4[g_idx, offset + t_idx] += coeff
                        
            offset += 1
            
    return U_4, V_4, W_4

def _schoolbook_4x4_4x3():
    """
    Returns U, V, W for 4x4x3 matrix multiplication with rank 48.
    Handles columns 4,5,6 of B.
    """
    U = np.zeros((16, 48))
    V = np.zeros((16, 48)) # B is 4x7, so V is (4*7, rank) = (28, rank). Wait.
    # V shape should be (m*p, rank) = (4*7, rank) = (28, rank).
    # In the 4x4x4 part, V was (16, 49). But global V is (28, rank).
    # So we need to return V with shape (28, rank) or (16, rank) and pad?
    # The interface expects V of shape (m*p, rank) = (28, rank).
    # So _strassen_4x4_4x4 should return V of shape (16, 49) which corresponds to B cols 0-3.
    # We need to place these into the correct rows of the global V (size 28).
    # Rows 0-15 correspond to B cols 0-3?
    # B index j*7+k.
    # j=0, k=0..3 -> 0,1,2,3.
    # j=0, k=4..6 -> 4,5,6.
    # j=1, k=0..3 -> 7,8,9,10.
    # ...
    # So B cols 0-3 are NOT contiguous in the global index.
    # j*7+k.
    # k=0: 0, 7, 14, 21.
    # k=1: 1, 8, 15, 22.
    # k=2: 2, 9, 16, 23.
    # k=3: 3, 10, 17, 24.
    # k=4: 4, 11, 18, 25.
    # k=5: 5, 12, 19, 26.
    # k=6: 6, 13, 20, 27.
    
    # My B_map in Strassen assumed contiguous blocks.
    # B00: j=0,1; k=0,1.
    # Indices:
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=1, k=0 -> 7
    # j=1, k=1 -> 8
    # So B00 indices are [0, 1, 7, 8].
    
    # I need to correct B_map.
    
    return U, V, W

# Corrected B_map
B_map_corrected = {
    0: [0, 1, 7, 8],   # B00: j=0,1; k=0,1
    1: [2, 3, 9, 10],  # B01: j=0,1; k=2,3
    2: [14, 15, 21, 22], # B10: j=2,3; k=0,1
    3: [16, 17, 23, 24]  # B11: j=2,3; k=2,3
}

# Corrected C_map (C is 4x4, index k*4+i)
# C00: k=0,1; i=0,1.
# k=0, i=0 -> 0
# k=0, i=1 -> 1
# k=1, i=0 -> 4
# k=1, i=1 -> 5
# Indices: [0, 1, 4, 5]
# C01: k=2,3; i=0,1.
# k=2, i=0 -> 2
# k=2, i=1 -> 3
# k=3, i=0 -> 6
# k=3, i=1 -> 7
# Indices: [2, 3, 6, 7]
# C10: k=0,1; i=2,3.
# k=0, i=2 -> 2
# k=0, i=3 -> 3
# k=1, i=2 -> 6
# k=1, i=3 -> 7
# Wait. C index is k*n+i. n=4.
# C10: i=2,3. k=0,1.
# k=0, i=2 -> 2
# k=0, i=3 -> 3
# k=1, i=2 -> 6
# k=1, i=3 -> 7
# Indices: [2, 3, 6, 7]
# This is the same as C01?
# C01: i=0,1. k=2,3.
# k=2, i=0 -> 2
# k=2, i=1 -> 3
# k=3, i=0 -> 6
# k=3, i=1 -> 7
# Yes, C01 and C10 have the same indices?
# No.
# C01: Column 1 of C? No, C is 4x4.
# C_{ij}. i is row, j is col.
# Index is j*n+i.
# C01: i=0,1. j=2,3.
# j=2, i=0 -> 2
# j=2, i=1 -> 3
# j=3, i=0 -> 6
# j=3, i=1 -> 7
# C10: i=2,3. j=0,1.
# j=0, i=2 -> 2
# j=0, i=3 -> 3
# j=1, i=2 -> 6
# j=1, i=3 -> 7
# Yes, the indices are the same!
# This implies C01 and C10 occupy the same positions in the flattened vector?
# No, C is 4x4. 16 entries.
# C00: 0,1,4,5
# C01: 2,3,6,7
# C10: 2,3,6,7 ??
# Let's re-verify C index: k*n+i.
# k is col, i is row.
# C01: col 1, rows 0,1.
# k=1, i=0 -> 1*4+0 = 4.
# k=1, i=1 -> 1*4+1 = 5.
# k=2, i=0 -> 2*4+0 = 8.
# k=2, i=1 -> 2*4+1 = 9.
# My previous mapping was wrong.
# C00: k=0,1; i=0,1.
# k=0, i=0 -> 0
# k=0, i=1 -> 1
# k=1, i=0 -> 4
# k=1, i=1 -> 5
# Indices: [0, 1, 4, 5]
# C01: k=2,3; i=0,1.
# k=2, i=0 -> 8
# k=2, i=1 -> 9
# k=3, i=0 -> 12
# k=3, i=1 -> 13
# Indices: [8, 9, 12, 13]
# C10: k=0,1; i=2,3.
# k=0, i=2 -> 2
# k=0, i=3 -> 3
# k=1, i=2 -> 6
# k=1, i=3 -> 7
# Indices: [2, 3, 6, 7]
# C11: k=2,3; i=2,3.
# k=2, i=2 -> 10
# k=2, i=3 -> 11
# k=3, i=2 -> 14
# k=3, i=3 -> 15
# Indices: [10, 11, 14, 15]

C_map_corrected = {
    0: [0, 1, 4, 5],
    1: [8, 9, 12, 13],
    2: [2, 3, 6, 7],
    3: [10, 11, 14, 15]
}

# A_map is correct.
A_map_corrected = {
    0: [0, 1, 4, 5],
    1: [2, 3, 6, 7],
    2: [8, 9, 12, 13],
    3: [10, 11, 14, 15]
}

# Now implement construct()
def construct():
    n, m, p = 4, 4, 7
    nm, mp, pn = 16, 28, 28
    
    # Part 1: Strassen 4x4x4 (Rank 49)
    U1, V1, W1 = _strassen_4x4_4x4()
    # V1 is (16, 49). We need to map to (28, 49).
    V1_global = np.zeros((28, 49))
    for i in range(16):
        # i is local B index. We need to map to global.
        # Local B index i corresponds to which global index?
        # In _strassen_4x4_4x4, V1 rows correspond to B_map indices.
        # B_map[0] = [0, 1, 7, 8] etc.
        # So V1[0, r] corresponds to B global index 0?
        # No, V1[0, r] is the coefficient for B_local[0].
        # B_local[0] is the (0,0) entry of B00 block.
        # Which is B[0,0]. Global index 0.
        # B_local[1] is (0,1) of B00. B[0,1]. Global index 1.
        # B_local[2] is (1,0) of B00. B[1,0]. Global index 7.
        # B_local[3] is (1,1) of B00. B[1,1]. Global index 8.
        # So V1[i, r] should be added to V1_global[global_idx, r].
        # We need a mapping from local B index (0..15) to global B index.
        
        # Local B index 0..3 -> B00
        # Local B index 4..7 -> B01
        # Local B index 8..11 -> B10
        # Local B index 12..15 -> B11
        
        # B00 local 0..3 -> global 0,1,7,8
        # B01 local 0..3 -> global 2,3,9,10
        # B10 local 0..3 -> global 14,15,21,22
        # B11 local 0..3 -> global 16,17,23,24
        
        block_id = i // 4
        local_in_block = i % 4
        
        if block_id == 0:
            g_idx = [0, 1, 7, 8][local_in_block]
        elif block_id == 1:
            g_idx = [2, 3, 9, 10][local_in_block]
        elif block_id == 2:
            g_idx = [14, 15, 21, 22][local_in_block]
        elif block_id == 3:
            g_idx = [16, 17, 23, 24][local_in_block]
            
        V1_global[g_idx, :] = V1[i, :]
        
    # Part 2: Schoolbook 4x4x3 (Rank 48)
    U2 = np.zeros((16, 48))
    V2_global = np.zeros((28, 48))
    W2 = np.zeros((16, 48))
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                k_global = k + 4 # cols 4,5,6
                a_idx = i * 4 + j
                b_idx = j * 7 + k_global
                c_idx = k_global * 4 + i
                
                U2[a_idx, col] = 1
                V2_global[b_idx, col] = 1
                W2[c_idx, col] = 1
                col += 1
                
    U_total = np.hstack([U1, U2])
    V_total = np.hstack([V1_global, V2_global])
    W_total = np.hstack([W1, W2])
    
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
