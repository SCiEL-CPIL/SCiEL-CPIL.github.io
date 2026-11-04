import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def get_strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm (rank 7).
    Indices:
    U: (2, 7) - A indices (row)
    V: (2, 7) - B indices (col/row) -- Wait, standard is A(i,j) B(j,k).
    Let's align with the task contract:
    U: (n*m, rank) -> (4, rank) for 2x2? No, n=2, m=2. U is (4, rank).
    V: (m*p, rank) -> (4, rank).
    W: (p*n, rank) -> (4, rank).
    
    Wait, the task contract says:
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    
    For 2x2x2:
    n=2, m=2, p=2.
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    
    The tensor is T[i, j, k] where i is row of A, j is col of A (row of B), k is col of B.
    U index: i*m + j? No.
    Usually U is indexed by (i, j) of A?
    Let's check the schoolbook seed:
    U[i * m + j][column] = 1
    V[j * p + k][column] = 1
    W[k * n + i][column] = 1
    
    So:
    U row index: i * m + j  (i in 0..n-1, j in 0..m-1)
    V row index: j * p + k  (j in 0..m-1, k in 0..p-1)
    W row index: k * n + i  (k in 0..p-1, i in 0..n-1)
    
    For 2x2x2:
    i, j, k in {0, 1}.
    U rows: 0..3 (00, 01, 10, 11)
    V rows: 0..3 (00, 01, 10, 11)
    W rows: 0..3 (00, 01, 10, 11)
    
    Strassen terms:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    We need to construct U, V, W such that:
    Sum_k U[i*m+j, k] * V[j*p+k, k] * W[k*n+i, k] = C[k, i] ?
    Wait, W index is k*n+i.
    For 2x2x2, W row index is k*2 + i.
    
    Let's map the Strassen coefficients to U, V, W.
    Each M_x is a product of linear forms in A and B.
    M_x = (sum_{i,j} u_{ij} A_{ij}) * (sum_{j,k} v_{jk} B_{jk})
    This M_x contributes to C_{k,i} with coefficient w_{ki}.
    
    So for each x in 1..7:
    We have a vector u_x of size 4 (for A entries).
    We have a vector v_x of size 4 (for B entries).
    We have a vector w_x of size 4 (for C entries).
    
    U[:, col] = u_x
    V[:, col] = v_x
    W[:, col] = w_x
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Indices:
    # A: 00=0, 01=1, 10=2, 11=3
    # B: 00=0, 01=1, 10=2, 11=3
    # C: 00=0, 01=1, 10=2, 11=3 (Note: W index is k*n+i. 
    #    C_{0,0} -> k=0, i=0 -> 0*2+0=0
    #    C_{0,1} -> k=1, i=0 -> 1*2+0=2
    #    C_{1,0} -> k=0, i=1 -> 0*2+1=1
    #    C_{1,1} -> k=1, i=1 -> 1*2+1=3
    #    So W indices: 0->C00, 1->C10, 2->C01, 3->C11
    
    # M1: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1  # C00
    W[3, 0] = 1  # C11
    
    # M2: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1  # C10
    
    # M3: A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1  # C01
    
    # M4: A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1  # C00
    W[2, 3] = 1  # C10
    
    # M5: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1 # C00
    W[2, 4] = 1  # C01
    
    # M6: (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C11
    
    # M7: (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # C00
    W[3, 6] = 1  # C11
    
    return U, V, W

def get_strassen_4x4x4():
    """
    Returns U, V, W for 4x4x4 matrix multiplication using Strassen (rank 49).
    Composes 7 blocks of 2x2x2 Strassen.
    """
    U2, V2, W2 = get_strassen_2x2x2()
    
    # 4x4x4: n=4, m=4, p=4.
    # U shape: (16, 49)
    # V shape: (16, 49)
    # W shape: (16, 49)
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # The 7 Strassen terms for 4x4x4:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Each Mx is a 2x2x2 multiplication of blocks.
    # Block A_ij is 2x2. Block B_jk is 2x2.
    # The "inputs" to the 2x2x2 Strassen are linear combinations of the 2x2 blocks.
    
    # We need to map the 2x2x2 Strassen coefficients to the 4x4x4 indices.
    
    # A indices (16):
    # 0: A00 (top-left 2x2)
    # 1: A01 (top-right 2x2)
    # 2: A10 (bottom-left 2x2)
    # 3: A11 (bottom-right 2x2)
    # Each block has 4 entries.
    # A00 entries: 0,1,4,5 (row 0: 0,1; row 1: 4,5)
    # A01 entries: 2,3,6,7
    # A10 entries: 8,9,12,13
    # A11 entries: 10,11,14,15
    
    # B indices (16):
    # B is 4x4.
    # B00 (top-left): 0,1,4,5
    # B01 (top-right): 2,3,6,7
    # B10 (bottom-left): 8,9,12,13
    # B11 (bottom-right): 10,11,14,15
    
    # C indices (16):
    # C00 (top-left): 0,1,4,5
    # C01 (top-right): 2,3,6,7
    # C10 (bottom-left): 8,9,12,13
    # C11 (bottom-right): 10,11,14,15
    
    # For each of the 7 Strassen terms, we have a 2x2x2 multiplication.
    # The 2x2x2 multiplication has 7 rank-1 components (from U2, V2, W2).
    # Total 49 components.
    
    # We define the linear combinations of blocks for each Mx.
    # Then for each of the 7 sub-components of the 2x2x2 multiplication,
    # we apply the coefficients from U2, V2, W2 to the specific block entries.
    
    # Block mapping for A:
    # Block 0 (A11 in Strassen formula? No, let's use 1-based for blocks)
    # Let's use 0-based: A00, A01, A10, A11.
    # Strassen M1 uses A00+A11.
    
    # We will iterate through the 7 Strassen terms.
    # For each term, we identify the linear combination of A blocks and B blocks.
    # Then we iterate through the 7 rank-1 components of the 2x2x2 Strassen.
    
    # U2, V2, W2 are for 2x2x2.
    # U2[i*m+j, k] where i,j are 0,1.
    # The "A" in 2x2x2 is a 2x2 matrix.
    # In 4x4x4, the "A" is a sum of 2x2 blocks.
    # So if U2[0, 0] = 1, it means "A_00" (top-left of the 2x2 block) is used.
    # This corresponds to the top-left entry of the sum of blocks.
    
    # Let's define the block coefficients for each Strassen term.
    # term: (coeffs_A, coeffs_B, coeffs_C)
    # coeffs_A: dict {block_idx: coeff}
    
    terms = [
        # M1: (A00+A11)(B00+B11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}), # C00, C11
        # M2: (A10+A11)B00
        ({2: 1, 3: 1}, {0: 1}, {2: 1}), # C10
        # M3: A00(B01-B11)
        ({0: 1}, {1: 1, 3: -1}, {1: 1}), # C01
        # M4: A11(B10-B00)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}), # C00, C10
        # M5: (A00+A01)B11
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}), # C00, C01
        # M6: (A10-A00)(B00+B01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}), # C11
        # M7: (A01-A11)(B10+B11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1, 3: 1}) # C00, C11
    ]
    
    # Block entry offsets:
    # Block 0 (top-left): rows 0-1, cols 0-1. Indices: 0,1,4,5
    # Block 1 (top-right): rows 0-1, cols 2-3. Indices: 2,3,6,7
    # Block 2 (bottom-left): rows 2-3, cols 0-1. Indices: 8,9,12,13
    # Block 3 (bottom-right): rows 2-3, cols 2-3. Indices: 10,11,14,15
    
    block_offsets = [
        [0, 1, 4, 5],
        [2, 3, 6, 7],
        [8, 9, 12, 13],
        [10, 11, 14, 15]
    ]
    
    col = 0
    for t_idx, (ca, cb, cc) in enumerate(terms):
        # For each of the 7 rank-1 components of the 2x2x2 Strassen
        for k in range(7):
            # Get the coefficients for the 2x2x2 sub-problem
            # U2[:, k], V2[:, k], W2[:, k]
            
            # U2 is 4x7. Rows 0..3 correspond to A entries 00, 01, 10, 11.
            # V2 is 4x7. Rows 0..3 correspond to B entries 00, 01, 10, 11.
            # W2 is 4x7. Rows 0..3 correspond to C entries 00, 10, 01, 11.
            
            # We need to map these to the 4x4x4 indices.
            
            # For U (A side):
            # U2[0, k] corresponds to A_00 (top-left of the 2x2 block).
            # This entry is at index 0 within the block.
            # So we add U2[0, k] * ca[block_idx] to U[offset, col].
            
            for a_idx in range(4):
                u_val = U2[a_idx, k]
                if u_val == 0: continue
                # a_idx 0->A00, 1->A01, 2->A10, 3->A11 in the 2x2 sub-problem
                # Which block does this correspond to?
                # The 2x2 sub-problem operates on the "blocks".
                # A_00 in sub-problem is the top-left block of the 4x4 matrix.
                # So it corresponds to block 0.
                # A_01 in sub-problem is the top-right block. Block 1.
                # A_10 in sub-problem is the bottom-left block. Block 2.
                # A_11 in sub-problem is the bottom-right block. Block 3.
                
                block_idx = a_idx
                if block_idx not in ca: continue
                coeff = ca[block_idx]
                
                # Which entry within the block?
                # a_idx 0 -> entry 0 (top-left of block)
                # a_idx 1 -> entry 1 (top-right of block)
                # a_idx 2 -> entry 2 (bottom-left of block)
                # a_idx 3 -> entry 3 (bottom-right of block)
                
                # Wait, U2 indices:
                # 0: i=0, j=0
                # 1: i=0, j=1
                # 2: i=1, j=0
                # 3: i=1, j=1
                
                # So a_idx maps directly to the entry index within the 2x2 block.
                # Entry 0: top-left. Entry 1: top-right. Entry 2: bottom-left. Entry 3: bottom-right.
                
                entry_idx = a_idx
                global_idx = block_offsets[block_idx][entry_idx]
                U[global_idx, col] += u_val * coeff
            
            # For V (B side):
            for b_idx in range(4):
                v_val = V2[b_idx, k]
                if v_val == 0: continue
                block_idx = b_idx
                if block_idx not in cb: continue
                coeff = cb[block_idx]
                entry_idx = b_idx
                global_idx = block_offsets[block_idx][entry_idx]
                V[global_idx, col] += v_val * coeff
            
            # For W (C side):
            # W2 indices:
            # 0: k=0, i=0 -> C00 (top-left)
            # 1: k=0, i=1 -> C10 (bottom-left)
            # 2: k=1, i=0 -> C01 (top-right)
            # 3: k=1, i=1 -> C11 (bottom-right)
            
            # cc dict keys: 0->C00, 1->C01, 2->C10, 3->C11
            # W2 row 0 -> C00. cc key 0.
            # W2 row 1 -> C10. cc key 2.
            # W2 row 2 -> C01. cc key 1.
            # W2 row 3 -> C11. cc key 3.
            
            w_map = {0: 0, 1: 2, 2: 1, 3: 3}
            
            for w_idx in range(4):
                w_val = W2[w_idx, k]
                if w_val == 0: continue
                cc_key = w_map[w_idx]
                if cc_key not in cc: continue
                coeff = cc[cc_key]
                
                # w_idx maps to entry within the 2x2 block of C.
                # w_idx 0 -> C00 (top-left). Entry 0.
                # w_idx 1 -> C10 (bottom-left). Entry 2.
                # w_idx 2 -> C01 (top-right). Entry 1.
                # w_idx 3 -> C11 (bottom-right). Entry 3.
                
                # Block indices for C:
                # 0: C00
                # 1: C01
                # 2: C10
                # 3: C11
                
                block_idx = cc_key
                entry_idx = w_idx
                global_idx = block_offsets[block_idx][entry_idx]
                W[global_idx, col] += w_val * coeff
            
            col += 1
            
    return U, V, W

def get_schoolbook_4x4x3():
    """
    Returns U, V, W for 4x4x3 matrix multiplication using Schoolbook (rank 12).
    """
    n, m, p = 4, 4, 3
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Get 4x4x4 Strassen (Rank 49)
    U1, V1, W1 = get_strassen_4x4x4()
    
    # Get 4x4x3 Schoolbook (Rank 12)
    U2, V2, W2 = get_schoolbook_4x4x3()
    
    # Concatenate
    # U1: (16, 49), U2: (16, 12)
    U = np.hstack([U1, U2])
    
    # V1: (16, 49), V2: (12, 12)?
    # Wait. V for 4x4x4 is (m*p, rank) = (4*4, 49) = (16, 49).
    # V for 4x4x3 is (m*p, rank) = (4*3, 12) = (12, 12).
    # We need to pad V2 to match V1's row count (16).
    # The 4x4x3 multiplication uses B of size 4x3.
    # The 4x4x4 multiplication uses B of size 4x4.
    # The total B is 4x7.
    # We split B into B1 (4x4) and B2 (4x3).
    # V1 corresponds to B1. Rows 0..15 (4*4).
    # V2 corresponds to B2. Rows 0..11 (4*3).
    # We need to map V2's rows to the global B indices.
    # Global B indices:
    # B1 (cols 0-3): indices 0..15.
    # B2 (cols 4-6): indices 16..27.
    
    # So we need to create a V_full of size (28, 61).
    # V1 goes into rows 0..15.
    # V2 goes into rows 16..27.
    
    V_full = np.zeros((28, 61))
    V_full[0:16, 0:49] = V1
    V_full[16:28, 49:61] = V2
    
    # W1: (16, 49), W2: (12, 12)?
    # W for 4x4x4 is (p*n, rank) = (4*4, 49) = (16, 49).
    # W for 4x4x3 is (p*n, rank) = (3*4, 12) = (12, 12).
    # Global C is 4x7.
    # C1 (cols 0-3): indices 0..15.
    # C2 (cols 4-6): indices 16..27.
    
    W_full = np.zeros((28, 61))
    W_full[0:16, 0:49] = W1
    W_full[16:28, 49:61] = W2
    
    # U is (16, 61).
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
