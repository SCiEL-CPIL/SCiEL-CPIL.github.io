import numpy as np
import time

def construct():
    n, m, p = 4, 4, 5
    
    # Rank 65 decomposition
    # 49 terms for 4x4x4 part (k=0..3)
    # 16 terms for 4x4x1 part (k=4)
    
    U = []
    V = []
    W = []
    
    # Helper to add a term
    def add_term(u, v, w):
        U.append(u)
        V.append(v)
        W.append(w)
        
    # 1. Strassen 4x4x4 part
    # We need to map the 2x2x2 Strassen terms to the 4x4 indices.
    
    # Define the 2x2x2 Strassen terms (u, v, w) as defined above
    # Indices for 2x2: 0:00, 1:01, 2:10, 3:11
    s_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # M3
        ([0, 0, 0, 1], [0, 0, 1, -1], [0, 0, 1, 1]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # M7
    ]
    
    # Mapping from 2x2 block index (0..3) to 4x4 indices
    # Block 0 (00): 0, 1, 4, 5
    # Block 1 (01): 2, 3, 6, 7
    # Block 2 (10): 8, 9, 12, 13
    # Block 3 (11): 10, 11, 14, 15
    block_indices = {
        0: [0, 1, 4, 5],
        1: [2, 3, 6, 7],
        2: [8, 9, 12, 13],
        3: [10, 11, 14, 15]
    }
    
    # We iterate over the 7 high-level Strassen terms
    # For each high-level term, we have linear combinations of A blocks and B blocks
    # E.g., M1 = (A00 + A11) * (B00 + B11)
    # This expands to 4 products of blocks: A00B00, A00B11, A11B00, A11B11
    # Each block product is a 2x2x2 multiplication, which expands to 7 rank-1 terms.
    # So M1 expands to 4 * 7 = 28 terms?
    # No, the Strassen algorithm is a bilinear algorithm.
    # The term M1 in the 4x4 case IS a product of two 2x2 matrices (the sums).
    # So we just take the Strassen 2x2x2 decomposition for the "inner" 2x2 matrices.
    # The "inner" A matrix is $A_{inner} = A_{00} + A_{11}$.
    # The "inner" B matrix is $B_{inner} = B_{00} + B_{11}$.
    # The product is $C_{inner} = A_{inner} B_{inner}$.
    # We decompose this product using the 7 rank-1 terms of Strassen 2x2x2.
    # Each of these 7 terms has u, v, w vectors of length 4 (for the 2x2 blocks).
    # We need to map these back to the 4x4 indices.
    
    # For a high-level term t (0..6):
    # u_t is a vector of length 4 (coeffs for A blocks 0..3)
    # v_t is a vector of length 4 (coeffs for B blocks 0..3)
    # w_t is a vector of length 4 (coeffs for C blocks 0..3)
    
    # For each high-level term t:
    #   For each inner Strassen term s (0..6):
    #     u_s is length 4, v_s is length 4, w_s is length 4.
    #     The contribution to the 4x4 U vector is:
    #     For each block b where u_t[b] != 0 and u_s[b'] != 0?
    #     No, $A_{inner}$ is a linear combination of A blocks.
    #     $A_{inner} = \sum_{b} u_t[b] A_b$.
    #     The inner Strassen term uses $u_s$ to select entries from $A_{inner}$.
    #     So the total coefficient for $A_{block\_b, entry\_e}$ is $u_t[b] \cdot u_s[e]$.
    #     Similarly for V and W.
    
    for t in range(7):
        u_t, v_t, w_t = s_terms[t]
        for s in range(7):
            u_s, v_s, w_s = s_terms[s]
            
            u_4x4 = [0.0] * 16
            v_4x4 = [0.0] * 16
            w_4x4 = [0.0] * 20 # p*n = 5*4 = 20
            
            # Fill U
            for b in range(4):
                if u_t[b] != 0:
                    for e in range(4):
                        if u_s[e] != 0:
                            idx_4x4 = block_indices[b][e]
                            u_4x4[idx_4x4] += u_t[b] * u_s[e]
                            
            # Fill V
            for b in range(4):
                if v_t[b] != 0:
                    for e in range(4):
                        if v_s[e] != 0:
                            idx_4x4 = block_indices[b][e]
                            v_4x4[idx_4x4] += v_t[b] * v_s[e]
                            
            # Fill W
            # C blocks correspond to k=0..3
            # W index is k*n + i.
            # Block b in C corresponds to rows i_b, cols k_b?
            # C is 4x5.
            # Block 0 (00) of C (4x4 part) corresponds to C[0:2, 0:2].
            # Indices in W:
            # C[0,0] -> 0*4+0 = 0
            # C[0,1] -> 1*4+0 = 4
            # C[1,0] -> 0*4+1 = 1
            # C[1,1] -> 1*4+1 = 5
            # So block 0 indices in W: 0, 4, 1, 5.
            # Block 1 (01) C[0:2, 2:4] -> k=2,3.
            # C[0,2] -> 2*4+0 = 8
            # C[0,3] -> 3*4+0 = 12
            # C[1,2] -> 2*4+1 = 9
            # C[1,3] -> 3*4+1 = 13
            # Block 2 (10) C[2:4, 0:2] -> k=0,1. i=2,3.
            # C[2,0] -> 0*4+2 = 2
            # C[2,1] -> 1*4+2 = 6
            # C[3,0] -> 0*4+3 = 3
            # C[3,1] -> 1*4+3 = 7
            # Block 3 (11) C[2:4, 2:4] -> k=2,3. i=2,3.
            # C[2,2] -> 2*4+2 = 10
            # C[2,3] -> 3*4+2 = 14
            # C[3,2] -> 2*4+3 = 11
            # C[3,3] -> 3*4+3 = 15
            
            c_block_indices = {
                0: [0, 4, 1, 5],
                1: [8, 12, 9, 13],
                2: [2, 6, 3, 7],
                3: [10, 14, 11, 15]
            }
            
            for b in range(4):
                if w_t[b] != 0:
                    for e in range(4):
                        if w_s[e] != 0:
                            idx_4x4 = c_block_indices[b][e]
                            w_4x4[idx_4x4] += w_t[b] * w_s[e]
                            
            add_term(u_4x4, v_4x4, w_4x4)
            
    # 2. Schoolbook for 5th column (k=4)
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    # 16 terms
    for i in range(4):
        for j in range(4):
            u_4x4 = [0.0] * 16
            v_4x4 = [0.0] * 20 # m*p = 4*5 = 20
            w_4x4 = [0.0] * 20
            
            # U index: i*m + j = i*4 + j
            u_4x4[i*4 + j] = 1.0
            
            # V index: j*p + k = j*5 + 4
            v_4x4[j*5 + 4] = 1.0
            
            # W index: k*n + i = 4*4 + i = 16 + i
            w_4x4[16 + i] = 1.0
            
            add_term(u_4x4, v_4x4, w_4x4)
            
    # Convert to lists
    U = [list(map(float, u)) for u in U]
    V = [list(map(float, v)) for v in V]
    W = [list(map(float, w)) for w in W]
    
    return {"U": U, "V": V, "W": W}
