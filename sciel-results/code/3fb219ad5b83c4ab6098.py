import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using 4 Strassen blocks
    # A (2x4) = [A0 (2x2), A1 (2x2)]
    # B (4x4) = [[B00 (2x2), B01 (2x2)],
    #            [B10 (2x2), B11 (2x2)]]
    # C = A0*B0 + A1*B1
    # B0 = [B00, B01], B1 = [B10, B11]
    
    # Strassen coefficients for 2x2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We will build U, V, W lists for each of the 4 blocks
    # Block 0: A0 * B00
    # Block 1: A0 * B01
    # Block 2: A1 * B10
    # Block 3: A1 * B11
    
    # U index: i*m + j. i in {0,1}, j in {0,1,2,3}
    # V index: j*p + k. j in {0,1,2,3}, k in {0,1,2,3}
    # W index: k*n + i. k in {0,1,2,3}, i in {0,1}
    
    U = [[0] * 28 for _ in range(8)]
    V = [[0] * 28 for _ in range(16)]
    W = [[0] * 28 for _ in range(8)]
    
    # Helper to add a Strassen term
    # block_a: 0 for A0, 1 for A1. A0 uses j=0,1. A1 uses j=2,3.
    # block_b: 0 for B00, 1 for B01, 2 for B10, 3 for B11.
    # B00 uses j=0,1 in V? No.
    # V index is j*p + k.
    # For A0 (i=0,1), j in 0,1.
    # For A1 (i=0,1), j in 2,3.
    
    # For B00, input j is 0,1. Output k is 0,1.
    # For B01, input j is 0,1. Output k is 2,3.
    # For B10, input j is 2,3. Output k is 0,1.
    # For B11, input j is 2,3. Output k is 2,3.
    
    # Strassen terms mapping:
    # A indices in 2x2: 00, 01, 10, 11 -> global i, j
    # B indices in 2x2: 00, 01, 10, 11 -> global j, k
    
    def add_strassen_term(term_id, a_offset_i, a_offset_j, b_offset_j, b_offset_k, coeff_a, coeff_b, coeff_w):
        # term_id: 0-6
        # This function is too complex to write generically with offsets.
        # I will hardcode the 28 terms.
        pass

    # Let's just list the 28 terms explicitly.
    # Term 0: M1 of A0*B00
    # A0: i=0,1; j=0,1.
    # B00: j=0,1; k=0,1.
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # U[0,0]=1, U[1,1]=1 (A[0,0], A[1,1])
    # V[0,0]=1, V[1,1]=1 (B[0,0], B[1,1])
    # W[0,0]=1, W[0,1]=1, W[1,0]=1, W[1,1]=1 (C[0,0], C[0,1], C[1,0], C[1,1])
    # Actually, W is for the output C.
    # C[0,0] gets M1. C[0,1] gets nothing?
    # M1 contributes to C[0,0] and C[1,1]?
    # In Strassen:
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # So for a specific block, the W vector has 4 non-zeros.
    
    # Block 0: A0*B00. Output C[0:2, 0:2].
    # Block 1: A0*B01. Output C[0:2, 2:4].
    # Block 2: A1*B10. Output C[0:2, 0:2]. (Adds to Block 0)
    # Block 3: A1*B11. Output C[0:2, 2:4]. (Adds to Block 1)
    
    # We need to sum the contributions.
    
    # Let's define the 7 Strassen terms for a generic 2x2x2.
    # Returns list of (U_vec, V_vec, W_vec) where vecs are length 2, 4, 4?
    # No, U is 2x2 (4 entries), V is 2x2 (4 entries), W is 2x2 (4 entries).
    
    # I will construct the 28 terms manually in code.
    
    # Term 0: Block 0, M1
    U[0][0] = 1; U[3][0] = 1 # A[0,0], A[1,1] (indices 0, 3 in 2x4? No. A[0,0] is idx 0. A[1,1] is idx 1*2+1=3)
    V[0][0] = 1; V[5][0] = 1 # B[0,0] idx 0. B[1,1] idx 1*4+1=5
    W[0][0] = 1; W[1][0] = 1; W[4][0] = 1; W[5][0] = 1 # C[0,0] idx 0. C[0,1] idx 1. C[1,0] idx 4. C[1,1] idx 5
    
    # This is getting tedious. I will write a helper to generate the 7 terms for a block and add them.
    
    def get_strassen_terms():
        # Returns 7 tuples of (u, v, w) for 2x2x2
        # u, v, w are dicts or lists of (index, coeff)
        terms = []
        # M1
        terms.append(([(0,1),(3,1)], [(0,1),(5,1)], [(0,1),(1,1),(4,1),(5,1)]))
        # M2: (A[1,0]+A[1,1]) * B[0,0]
        terms.append(([(2,1),(3,1)], [(0,1)], [(4,1),(5,1)])) # C[1,0], C[1,1]
        # M3: A[0,0] * (B[0,1]-B[1,1])
        terms.append(([(0,1)], [(1,1),(5,-1)], [(1,1),(5,1)])) # C[0,1], C[1,1]
        # M4: A[1,1] * (B[1,0]-B[0,0])
        terms.append(([(3,1)], [(4,1),(0,-1)], [(0,1),(4,1)])) # C[0,0], C[1,0]
        # M5: (A[0,0]+A[0,1]) * B[1,1]
        terms.append(([(0,1),(1,1)], [(5,1)], [(1,-1),(5,1)])) # C[0,1], C[1,1]
        # M6: (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
        terms.append(([(2,1),(0,-1)], [(0,1),(1,1)], [(5,1)])) # C[1,1]
        # M7: (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
        terms.append(([(1,1),(3,-1)], [(4,1),(5,1)], [(0,1),(1,1)])) # C[0,0], C[0,1]
        
        return terms

    strassen_terms = get_strassen_terms()
    
    # Block 0: A0 (j=0,1) * B00 (j=0,1, k=0,1)
    # U offset: 0. V offset: 0. W offset: 0.
    for t in range(7):
        u, v, w = strassen_terms[t]
        for idx, c in u:
            U[idx][t] += c
        for idx, c in v:
            V[idx][t] += c
        for idx, c in w:
            W[idx][t] += c
            
    # Block 1: A0 (j=0,1) * B01 (j=0,1, k=2,3)
    # U offset: 0. V offset: 0 (j is same). W offset: 2 (k starts at 2).
    # V index: j*4 + k. For B01, k is 2,3. So V indices are 0*4+2=2, 1*4+3=7?
    # Wait. B01 is the right half of the top row of B.
    # B is 4x4.
    # B00: rows 0,1; cols 0,1.
    # B01: rows 0,1; cols 2,3.
    # So for V, the row index j is 0,1. The col index k is 2,3.
    # V index = j*4 + k.
    # So for B01, V indices are 2, 3, 6, 7.
    # In Strassen, B indices are 0,1,2,3 corresponding to (0,0),(0,1),(1,0),(1,1).
    # We need to map Strassen B indices to global V indices.
    # Strassen B[0,0] -> Global B[0,2] -> V idx 2.
    # Strassen B[0,1] -> Global B[0,3] -> V idx 3.
    # Strassen B[1,0] -> Global B[1,2] -> V idx 6.
    # Strassen B[1,1] -> Global B[1,3] -> V idx 7.
    
    # W offset: k starts at 2.
    # W index = k*2 + i.
    # Strassen C[0,0] -> Global C[0,2] -> W idx 2*2+0=4.
    # Strassen C[0,1] -> Global C[0,3] -> W idx 2*2+1=5.
    # Strassen C[1,0] -> Global C[1,2] -> W idx 3*2+0=6.
    # Strassen C[1,1] -> Global C[1,3] -> W idx 3*2+1=7.
    
    v_map = {0:2, 1:3, 2:6, 3:7}
    w_map = {0:4, 1:5, 2:6, 3:7}
    
    for t in range(7):
        u, v, w = strassen_terms[t]
        col = t + 7
        for idx, c in u:
            U[idx][col] += c
        for idx, c in v:
            V[v_map[idx]][col] += c
        for idx, c in w:
            W[w_map[idx]][col] += c
            
    # Block 2: A1 (j=2,3) * B10 (j=2,3, k=0,1)
    # U offset: j starts at 2.
    # Strassen A[0,0] -> Global A[0,2] -> U idx 0*4+2=2.
    # Strassen A[0,1] -> Global A[0,3] -> U idx 0*4+3=3.
    # Strassen A[1,0] -> Global A[1,2] -> U idx 1*4+2=6.
    # Strassen A[1,1] -> Global A[1,3] -> U idx 1*4+3=7.
    u_map = {0:2, 1:3, 2:6, 3:7}
    
    # V offset: j starts at 2.
    # Strassen B[0,0] -> Global B[2,0] -> V idx 2*4+0=8.
    # Strassen B[0,1] -> Global B[2,1] -> V idx 2*4+1=9.
    # Strassen B[1,0] -> Global B[3,0] -> V idx 3*4+0=12.
    # Strassen B[1,1] -> Global B[3,1] -> V idx 3*4+1=13.
    v_map = {0:8, 1:9, 2:12, 3:13}
    
    # W offset: k starts at 0.
    # Same as Block 0 W mapping.
    w_map = {0:0, 1:1, 2:4, 3:5}
    
    for t in range(7):
        u, v, w = strassen_terms[t]
        col = t + 14
        for idx, c in u:
            U[u_map[idx]][col] += c
        for idx, c in v:
            V[v_map[idx]][col] += c
        for idx, c in w:
            W[w_map[idx]][col] += c
            
    # Block 3: A1 (j=2,3) * B11 (j=2,3, k=2,3)
    u_map = {0:2, 1:3, 2:6, 3:7}
    v_map = {0:10, 1:11, 2:14, 3:15} # B[2,2] idx 10, B[2,3] idx 11, B[3,2] idx 14, B[3,3] idx 15
    w_map = {0:4, 1:5, 2:6, 3:7}
    
    for t in range(7):
        u, v, w = strassen_terms[t]
        col = t + 21
        for idx, c in u:
            U[u_map[idx]][col] += c
        for idx, c in v:
            V[v_map[idx]][col] += c
        for idx, c in w:
            W[w_map[idx]][col] += c
            
    # Convert to numpy for verification (optional, but good for debugging)
    # We must return lists of lists.
    
    return {"U": U, "V": V, "W": W}
