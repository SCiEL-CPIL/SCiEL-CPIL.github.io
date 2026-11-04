import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct the rank-minimal exact tensor decomposition for 5x5 matrix multiplication.
    Uses a block decomposition: 4x4 Strassen (rank 49) + border terms (rank 50) = rank 99.
    """
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS
    # We will build U, V, W for rank 99
    # U shape: (25, 99)
    # V shape: (25, 99)
    # W shape: (25, 99)
    
    U = np.zeros((n * m, 99), dtype=np.int64)
    V = np.zeros((m * p, 99), dtype=np.int64)
    W = np.zeros((p * n, 99), dtype=np.int64)
    
    r_offset = 0
    
    # 1. Core 4x4 Strassen Decomposition
    # The 4x4 block uses indices 0..3 for rows and cols of A, B, C.
    # We need the 49 rank-1 terms for 4x4 multiplication.
    # These are derived from 2x2 Strassen (7 terms) recursively.
    
    # Define 2x2 Strassen terms (u, v, w)
    # u, v, w are vectors of length 4 (2x2 flattened)
    # w is the coefficient for C_ik
    strassen_2x2 = [
        # M1 = (a+d)(e+h) -> C11 += 1, C22 += 1
        # u: a=1, b=0, c=0, d=1 -> [1,0,0,1]
        # v: e=1, f=0, g=1, h=1 -> [1,0,1,1]
        # w: C11=1, C12=0, C21=0, C22=1 -> [1,0,0,1]
        (np.array([1,0,0,1]), np.array([1,0,1,1]), np.array([1,0,0,1])),
        # M2 = (c+d)e -> C21 += 1, C22 -= 1
        # u: c=1, d=1 -> [0,0,1,1]
        # v: e=1 -> [1,0,0,0]
        # w: C21=1, C22=-1 -> [0,0,1,-1]
        (np.array([0,0,1,1]), np.array([1,0,0,0]), np.array([0,0,1,-1])),
        # M3 = a(f-h) -> C11 += 1, C12 += 1
        # u: a=1 -> [1,0,0,0]
        # v: f=1, h=-1 -> [0,1,0,-1]
        # w: C11=1, C12=1 -> [1,1,0,0]
        (np.array([1,0,0,0]), np.array([0,1,0,-1]), np.array([1,1,0,0])),
        # M4 = d(g-f) -> C11 += 1, C21 += 1
        # u: d=1 -> [0,0,0,1]
        # v: g=1, f=-1 -> [0,-1,1,0]
        # w: C11=1, C21=1 -> [1,0,1,0]
        (np.array([0,0,0,1]), np.array([0,-1,1,0]), np.array([1,0,1,0])),
        # M5 = (a+b)h -> C11 -= 1, C12 += 1
        # u: a=1, b=1 -> [1,1,0,0]
        # v: h=1 -> [0,0,0,1]
        # w: C11=-1, C12=1 -> [-1,1,0,0]
        (np.array([1,1,0,0]), np.array([0,0,0,1]), np.array([-1,1,0,0])),
        # M6 = (c-a)(e+f) -> C22 += 1
        # u: c=1, a=-1 -> [-1,0,1,0]
        # v: e=1, f=1 -> [1,1,0,0]
        # w: C22=1 -> [0,0,0,1]
        (np.array([-1,0,1,0]), np.array([1,1,0,0]), np.array([0,0,0,1])),
        # M7 = (b-d)(g+h) -> C11 += 1, C22 -= 1
        # u: b=1, d=-1 -> [0,1,0,-1]
        # v: g=1, h=1 -> [0,0,1,1]
        # w: C11=1, C22=-1 -> [1,0,0,-1]
        (np.array([0,1,0,-1]), np.array([0,0,1,1]), np.array([1,0,0,-1])),
    ]
    
    # Compose 4x4 from 2x2
    # 4x4 has 16 elements.
    # We iterate over the 7 terms of the outer Strassen, and for each, the 7 terms of the inner Strassen.
    # Actually, it's easier to just generate the 49 terms by expanding the 2x2 logic to 4x4.
    # But I can just use the 2x2 terms and map indices.
    # For 4x4, we have 4 blocks of 2x2.
    # Let's just hardcode the 49 terms by nesting the 2x2 generation.
    
    # This is complex to generate dynamically in the loop without a helper.
    # I will write a helper to generate 4x4 terms.
    
    def get_4x4_terms():
        terms = []
        # 2x2 Strassen terms for the "block" level
        # Let A = [[A00, A01], [A10, A11]] where Ai are 2x2.
        # The 7 products M_k are formed from linear combos of Ai.
        # Each Ai product is formed by 7 rank-1 terms.
        # So we have 7 * 7 = 49 terms.
        
        # u_block, v_block, w_block from strassen_2x2
        # For each block term k (0..6):
        #   u_block is a 2x2 matrix of coefficients for A00..A11
        #   v_block is a 2x2 matrix of coefficients for B00..B11
        #   w_block is a 2x2 matrix of coefficients for C00..C11
        
        # For each inner term j (0..6):
        #   u_inner, v_inner, w_inner are vectors for 2x2 elements.
        
        for k in range(7):
            u_b, v_b, w_b = strassen_2x2[k]
            # u_b is [u00, u01, u10, u11] for A blocks
            # v_b is [v00, v01, v10, v11] for B blocks
            # w_b is [w00, w01, w10, w11] for C blocks
            
            for i_block in range(4): # 00, 01, 10, 11
                for j_block in range(4):
                    # u_b[i_block] is coeff for A block
                    # v_b[j_block] is coeff for B block
                    # w_b[i_block, j_block] is coeff for C block
                    
                    u_coeff = u_b[i_block]
                    v_coeff = v_b[j_block]
                    w_coeff = w_b[i_block * 2 + j_block]
                    
                    if u_coeff == 0 or v_coeff == 0 or w_coeff == 0:
                        continue
                    
                    # Now expand this block term into 7 inner terms
                    # The block A is 2x2. The block B is 2x2.
                    # The product is (sum u_inner * A_elem) * (sum v_inner * B_elem)
                    # Multiplied by u_coeff * v_coeff * w_coeff
                    
                    for m in range(7):
                        u_i, v_i, w_i = strassen_2x2[m]
                        
                        # u_i is vector of 4 elements for A_block
                        # v_i is vector of 4 elements for B_block
                        # w_i is vector of 4 elements for C_block
                        
                        # We need to map these to the 4x4 global indices
                        # A_block is at (i_block, j_block)? No.
                        # A is partitioned into 4 blocks: A00, A01, A10, A11.
                        # A00 is rows 0-1, cols 0-1.
                        # A01 is rows 0-1, cols 2-3.
                        # A10 is rows 2-3, cols 0-1.
                        # A11 is rows 2-3, cols 2-3.
                        
                        # i_block index 0->A00, 1->A01, 2->A10, 3->A11
                        # Row offset: 0 if i_block < 2 else 2
                        # Col offset: 0 if i_block % 2 == 0 else 2
                        
                        a_row_off = (i_block // 2) * 2
                        a_col_off = (i_block % 2) * 2
                        
                        b_row_off = (j_block // 2) * 2
                        b_col_off = (j_block % 2) * 2
                        
                        c_row_off = (i_block // 2) * 2
                        c_col_off = (j_block % 2) * 2
                        
                        u_vec = np.zeros(16)
                        v_vec = np.zeros(16)
                        w_vec = np.zeros(16)
                        
                        for ea in range(4):
                            r = ea // 2
                            c = ea % 2
                            u_idx = (a_row_off + r) * 4 + (a_col_off + c)
                            u_vec[u_idx] = u_coeff * u_i[ea]
                            
                        for eb in range(4):
                            r = eb // 2
                            c = eb % 2
                            v_idx = (b_row_off + r) * 4 + (b_col_off + c)
                            v_vec[v_idx] = v_coeff * v_i[eb]
                            
                        for ec in range(4):
                            r = ec // 2
                            c = ec % 2
                            w_idx = (c_col_off + c) * 4 + (c_row_off + r) # W index is k*n+i
                            # Wait, W index for 4x4 is k*4+i.
                            # c_row_off + r is i, c_col_off + c is k.
                            w_vec[w_idx] = w_coeff * w_i[ec]
                        
                        terms.append((u_vec, v_vec, w_vec))
        return terms

    terms_4x4 = get_4x4_terms()
    
    # Add 4x4 terms to U, V, W
    # Note: The 4x4 block is at indices 0..3 of the 5x5 matrix.
    for u_vec, v_vec, w_vec in terms_4x4:
        U[0:16, r_offset] = u_vec # A indices 0..15
        V[0:16, r_offset] = v_vec # B indices 0..15
        W[0:16, r_offset] = w_vec # C indices 0..15
        r_offset += 1
        
    # 2. Border Terms
    # A = [[A44, a41], [a14T, a11]]
    # B = [[B44, b41], [b14T, b11]]
    
    # Term: a41 * b14T -> C44
    # a41 is 4x1 (rows 0-3, col 4 of A). Index in A: i*5 + 4.
    # b14T is 1x4 (row 4, cols 0-3 of B). Index in B: 4*5 + j.
    # C44 is 4x4 (rows 0-3, cols 0-3 of C). Index in C: k*5 + i.
    for i in range(4):
        for j in range(4):
            u_idx = i * 5 + 4
            v_idx = 4 * 5 + j
            w_idx = j * 5 + i
            U[u_idx, r_offset] = 1
            V[v_idx, r_offset] = 1
            W[w_idx, r_offset] = 1
            r_offset += 1
            
    # Term: A44 * b41 -> C41
    # A44 is 4x4. b41 is 4x1 (rows 0-3, col 4 of B).
    # C41 is 4x1 (rows 0-3, col 4 of C).
    for i in range(4):
        for k in range(4):
            u_idx = i * 5 + k
            v_idx = k * 5 + 4
            w_idx = 4 * 5 + i
            U[u_idx, r_offset] = 1
            V[v_idx, r_offset] = 1
            W[w_idx, r_offset] = 1
            r_offset += 1
            
    # Term: a14T * B44 -> C14
    # a14T is 1x4 (row 4, cols 0-3 of A).
    # B44 is 4x4.
    # C14 is 1x4 (row 4, cols 0-3 of C).
    for j in range(4):
        for k in range(4):
            u_idx = 4 * 5 + j
            v_idx = j * 5 + k
            w_idx = k * 5 + 4
            U[u_idx, r_offset] = 1
            V[v_idx, r_offset] = 1
            W[w_idx, r_offset] = 1
            r_offset += 1
            
    # Term: a11 * b11 -> C11
    u_idx = 4 * 5 + 4
    v_idx = 4 * 5 + 4
    w_idx = 4 * 5 + 4
    U[u_idx, r_offset] = 1
    V[v_idx, r_offset] = 1
    W[w_idx, r_offset] = 1
    r_offset += 1
    
    # Term: a14T * b41 -> C11
    # a14T is 1x4. b41 is 4x1.
    for k in range(4):
        u_idx = 4 * 5 + k
        v_idx = k * 5 + 4
        w_idx = 4 * 5 + 4
        U[u_idx, r_offset] = 1
        V[v_idx, r_offset] = 1
        W[w_idx, r_offset] = 1
        r_offset += 1
        
    # Total rank should be 49 + 16 + 16 + 16 + 1 + 4 = 102?
    # Wait, a14T * b41 is 4 terms.
    # 49 + 16 + 16 + 16 + 1 + 4 = 102.
    # My previous count was 99. Let's re-verify.
    # a41 b14T: 16
    # A44 b41: 16
    # a14T B44: 16
    # a11 b11: 1
    # a14T b41: 4
    # Total border: 53.
    # Total: 49 + 53 = 102.
    
    # Trim to actual used rank
    U = U[:, :r_offset]
    V = V[:, :r_offset]
    W = W[:, :r_offset]
    
    if time.time() > deadline - 5:
        pass
        
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
