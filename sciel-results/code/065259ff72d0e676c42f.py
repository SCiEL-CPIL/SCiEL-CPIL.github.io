import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # We construct a decomposition of rank 28
    # Part 1: j=2 (M_{2,1,5}) -> Rank 10
    # Part 2: j=0,1 (M_{2,2,5}) -> Split into k=0,1 (Rank 7), k=2,3 (Rank 7), k=4 (Rank 4)
    # Total Rank = 10 + 7 + 7 + 4 = 28
    
    R = 28
    U = np.zeros((nm, R))
    V = np.zeros((mp, R))
    W = np.zeros((pn, R))
    
    r_idx = 0
    
    # --- Part 1: j=2 ---
    # A indices: i*3 + 2 for i in 0,1 -> 2, 5
    # B indices: 2*5 + k for k in 0..4 -> 10, 11, 12, 13, 14
    # C indices: k*2 + i for k in 0..4, i in 0,1
    for i in range(n):
        for k in range(p):
            a = i * m + 2
            b = 2 * p + k
            c = k * n + i
            
            U[a, r_idx] = 1
            V[b, r_idx] = 1
            W[c, r_idx] = 1
            r_idx += 1
            
    # --- Part 2: j=0,1 ---
    # We handle k=0,1 using Strassen (Rank 7)
    # We handle k=2,3 using Strassen (Rank 7)
    # We handle k=4 using Trivial (Rank 4)
    
    # Helper to add Strassen block for A (2x2), B (2x2), C (2x2)
    def add_strassen_block(r_start, a_offsets, b_offsets, c_offsets):
        # a_offsets: dict mapping local (i,j) to global a
        # b_offsets: dict mapping local (j,k) to global b
        # c_offsets: dict mapping local (k,i) to global c
        # Local indices: i in 0,1; j in 0,1; k in 0,1
        
        # M1 = (A11 + A22)(B11 + B22)
        U[a_offsets[(0,0)], r_start+0] += 1
        U[a_offsets[(1,1)], r_start+0] += 1
        V[b_offsets[(0,0)], r_start+0] += 1
        V[b_offsets[(1,1)], r_start+0] += 1
        W[c_offsets[(0,0)], r_start+0] += 1
        W[c_offsets[(1,1)], r_start+0] += 1
        W[c_offsets[(0,1)], r_start+0] -= 1 # Wait, C11 = M1 + M4 - M5 - M7
        # Let's be precise with signs.
        
        # It's easier to define the terms explicitly.
        # M1: A00 A11? No.
        # Standard Strassen:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22) B11
        # M3 = A11 (B21 - B11)
        # M4 = A22 (B12 - B22)
        # M5 = (A11 + A12) B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 + M4 - M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M7
        
        # Note: My local indices are 0-based.
        # A11 -> A[0,0]
        # A22 -> A[1,1]
        # etc.
        
        # Term 1: M1
        # A: (0,0) and (1,1)
        # B: (0,0) and (1,1)
        # C: (0,0) and (1,1)
        U[a_offsets[(0,0)], r_start+0] += 1
        U[a_offsets[(1,1)], r_start+0] += 1
        V[b_offsets[(0,0)], r_start+0] += 1
        V[b_offsets[(1,1)], r_start+0] += 1
        W[c_offsets[(0,0)], r_start+0] += 1
        W[c_offsets[(1,1)], r_start+0] += 1
        
        # Term 2: M2
        # A: (1,0) and (1,1)  (A21 + A22)
        # B: (0,0)            (B11)
        # C: (1,0)            (C21)
        U[a_offsets[(1,0)], r_start+1] += 1
        U[a_offsets[(1,1)], r_start+1] += 1
        V[b_offsets[(0,0)], r_start+1] += 1
        W[c_offsets[(1,0)], r_start+1] += 1
        
        # Term 3: M3
        # A: (0,0)            (A11)
        # B: (1,0) and (0,0)  (B21 - B11)
        # C: (0,1)            (C12)
        U[a_offsets[(0,0)], r_start+2] += 1
        V[b_offsets[(1,0)], r_start+2] += 1
        V[b_offsets[(0,0)], r_start+2] -= 1
        W[c_offsets[(0,1)], r_start+2] += 1
        
        # Term 4: M4
        # A: (1,1)            (A22)
        # B: (0,1) and (1,1)  (B12 - B22)
        # C: (0,0) and (1,0)  (C11 and C21)
        U[a_offsets[(1,1)], r_start+3] += 1
        V[b_offsets[(0,1)], r_start+3] += 1
        V[b_offsets[(1,1)], r_start+3] -= 1
        W[c_offsets[(0,0)], r_start+3] += 1
        W[c_offsets[(1,0)], r_start+3] += 1
        
        # Term 5: M5
        # A: (0,0) and (0,1)  (A11 + A12)
        # B: (1,1)            (B22)
        # C: (0,0) and (0,1)  (C11 and C12)
        U[a_offsets[(0,0)], r_start+4] += 1
        U[a_offsets[(0,1)], r_start+4] += 1
        V[b_offsets[(1,1)], r_start+4] += 1
        W[c_offsets[(0,0)], r_start+4] -= 1 # C11 = ... - M5
        W[c_offsets[(0,1)], r_start+4] += 1 # C12 = ... + M5
        
        # Term 6: M6
        # A: (1,0) and (0,0)  (A21 - A11)
        # B: (0,0) and (0,1)  (B11 + B12)
        # C: (1,1)            (C22)
        U[a_offsets[(1,0)], r_start+5] += 1
        U[a_offsets[(0,0)], r_start+5] -= 1
        V[b_offsets[(0,0)], r_start+5] += 1
        V[b_offsets[(0,1)], r_start+5] += 1
        W[c_offsets[(1,1)], r_start+5] += 1
        
        # Term 7: M7
        # A: (0,1) and (1,1)  (A12 - A22)
        # B: (1,0) and (1,1)  (B21 + B22)
        # C: (0,0) and (1,1)  (C11 and C22)
        U[a_offsets[(0,1)], r_start+6] += 1
        U[a_offsets[(1,1)], r_start+6] -= 1
        V[b_offsets[(1,0)], r_start+6] += 1
        V[b_offsets[(1,1)], r_start+6] += 1
        W[c_offsets[(0,0)], r_start+6] -= 1 # C11 = ... - M7
        W[c_offsets[(1,1)], r_start+6] += 1 # C22 = ... + M7

        return r_start + 7

    # Block 2a: k=0,1
    # A local (i,j) -> Global a = i*3 + j
    # B local (j,k) -> Global b = j*5 + k
    # C local (k,i) -> Global c = k*2 + i
    
    a_off_2a = {(0,0): 0, (0,1): 1, (1,0): 3, (1,1): 4}
    b_off_2a = {(0,0): 0, (0,1): 1, (1,0): 5, (1,1): 6}
    c_off_2a = {(0,0): 0, (0,1): 1, (1,0): 2, (1,1): 3}
    
    r_idx = add_strassen_block(r_idx, a_off_2a, b_off_2a, c_off_2a)
    
    # Block 2b: k=2,3
    a_off_2b = {(0,0): 0, (0,1): 1, (1,0): 3, (1,1): 4}
    b_off_2b = {(0,0): 2, (0,1): 3, (1,0): 7, (1,1): 8}
    c_off_2b = {(0,0): 4, (0,1): 5, (1,0): 6, (1,1): 7}
    
    r_idx = add_strassen_block(r_idx, a_off_2b, b_off_2b, c_off_2b)
    
    # Block 2c: k=4 (Trivial M_{2,2,1})
    # C00 = A00 B00 + A01 B10
    # C10 = A10 B00 + A11 B10
    # Global:
    # A00 -> 0, A01 -> 1, A10 -> 3, A11 -> 4
    # B00 (j=0, k=4) -> 4, B10 (j=1, k=4) -> 9
    # C00 (k=4, i=0) -> 8, C10 (k=4, i=1) -> 9
    
    # Term 1: A00 B00 C00
    U[0, r_idx] += 1
    V[4, r_idx] += 1
    W[8, r_idx] += 1
    r_idx += 1
    
    # Term 2: A01 B10 C00
    U[1, r_idx] += 1
    V[9, r_idx] += 1
    W[8, r_idx] += 1
    r_idx += 1
    
    # Term 3: A10 B00 C10
    U[3, r_idx] += 1
    V[4, r_idx] += 1
    W[9, r_idx] += 1
    r_idx += 1
    
    # Term 4: A11 B10 C10
    U[4, r_idx] += 1
    V[9, r_idx] += 1
    W[9, r_idx] += 1
    r_idx += 1
    
    # Verify r_idx is 28
    # 10 + 7 + 7 + 4 = 28
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
