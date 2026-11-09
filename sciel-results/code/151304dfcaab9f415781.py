import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will construct the decomposition for M_{2,5,5}
    # Split A (2x5) into A1 (2x2) and A2 (2x3)
    # Split B (5x5) into B1 (2x5) and B2 (3x5)
    # C = A1 B1 + A2 B2
    
    # Indices for A: i in 0..1, j in 0..4
    # Indices for B: j in 0..4, k in 0..4
    # Indices for C: k in 0..4, i in 0..1 (W[k*n+i])
    
    rank = 0
    U = []
    V = []
    W = []
    
    def add_term(u, v, w):
        nonlocal rank
        U.append(u)
        V.append(v)
        W.append(w)
        rank += 1

    # Part 1: A1 (2x2) * B1 (2x5)
    # A1 indices: i in 0..1, j in 0..1 (mapped to global j 0..1)
    # B1 indices: j in 0..1 (mapped to global j 0..1), k in 0..4
    
    # B1 columns 0,1: 2x2 block. Use Strassen (7 terms)
    # B1 columns 2,3: 2x2 block. Use Strassen (7 terms)
    # B1 column 4: 2x1 block. Use Schoolbook (4 terms)
    
    # Strassen terms for 2x2x2
    # P1 = A00 (B00 + B11)
    # P2 = (A00 + A01) B11
    # P3 = (A10 + A11) B00
    # P4 = A11 (B00 - B10)
    # P5 = (A00 + A11) (B10 + B11)
    # P6 = (A01 - A10 + A11) (B00 + B01)
    # P7 = (A00 + A01 - A10 - A11) (B10 + B11)
    
    # We need to map these to global indices.
    # A1[0,0] is global A[0,0]. A1[0,1] is global A[0,1].
    # A1[1,0] is global A[1,0]. A1[1,1] is global A[1,1].
    
    # B1[0,0] is global B[0,0]. B1[0,1] is global B[0,1].
    # B1[1,0] is global B[1,0]. B1[1,1] is global B[1,1].
    
    # For block k_offset:
    # B_local[0,0] is global B[0, k_offset+0]
    # B_local[0,1] is global B[0, k_offset+1]
    # B_local[1,0] is global B[1, k_offset+0]
    # B_local[1,1] is global B[1, k_offset+1]
    
    def add_strassen_block(k_offset):
        # Term 1: A00 * (B00 + B11)
        # U: A00 -> 1 at (0,0)
        # V: B00 + B11 -> 1 at (0, k_offset), 1 at (1, k_offset+1)
        # W: C00 + C01? No, W is output.
        # The term contributes to C00 and C01?
        # Strassen P1 contributes to C00 and C11.
        # C00 = P1 + P4 - P5 + P7
        # C01 = P2 + P5
        # C10 = P3 + P6
        # C11 = P1 - P2 + P3 + P6
        
        # We need to add 7 terms.
        # Each term is a product of a linear combo of A and a linear combo of B.
        # The result is added to specific C entries.
        
        # Term 1: A00 * (B00 + B11)
        # U: e_00
        # V: e_0,k + e_1,k+1
        # W: e_k,0 (for C00) + e_k+1,0 (for C01)? No.
        # P1 contributes to C00 and C11.
        # So W must have 1 at (k, 0) and (k+1, 1).
        u = [0]*10; u[0] = 1
        v = [0]*25; v[0*5 + k_offset] = 1; v[1*5 + k_offset+1] = 1
        w = [0]*10; w[k_offset*2 + 0] = 1; w[(k_offset+1)*2 + 1] = 1
        add_term(u, v, w)
        
        # Term 2: (A00 + A01) * B11
        # U: e_00 + e_01
        # V: e_1,k+1
        # W: C01 (k+1, 0) and C11 (k+1, 1)
        u = [0]*10; u[0] = 1; u[1] = 1
        v = [0]*25; v[1*5 + k_offset+1] = 1
        w = [0]*10; w[(k_offset+1)*2 + 0] = 1; w[(k_offset+1)*2 + 1] = 1
        add_term(u, v, w)
        
        # Term 3: (A10 + A11) * B00
        # U: e_10 + e_11
        # V: e_0,k
        # W: C10 (k, 1) and C11 (k+1, 1)
        u = [0]*10; u[4] = 1; u[5] = 1
        v = [0]*25; v[0*5 + k_offset] = 1
        w = [0]*10; w[k_offset*2 + 1] = 1; w[(k_offset+1)*2 + 1] = 1
        add_term(u, v, w)
        
        # Term 4: A11 * (B00 - B10)
        # U: e_11
        # V: e_0,k - e_1,k
        # W: C00 (k, 0) and C11 (k+1, 1)
        u = [0]*10; u[5] = 1
        v = [0]*25; v[0*5 + k_offset] = 1; v[1*5 + k_offset] = -1
        w = [0]*10; w[k_offset*2 + 0] = 1; w[(k_offset+1)*2 + 1] = 1
        add_term(u, v, w)
        
        # Term 5: (A00 + A11) * (B10 + B11)
        # U: e_00 + e_11
        # V: e_1,k + e_1,k+1
        # W: C00 (k, 0) - C01 (k+1, 0) + C10 (k, 1) - C11 (k+1, 1)
        # C00 = ... - P5
        # C01 = ... + P5
        # C10 = ... + P5? No, C10 = P3 + P6.
        # C11 = ... + P6?
        # Let's recheck Strassen formulas.
        # C00 = P1 + P4 - P5 + P7
        # C01 = P2 + P5
        # C10 = P3 + P6
        # C11 = P1 - P2 + P3 + P6
        
        # P5 = (A00 + A11)(B10 + B11)
        # Contributes to C00 (-1), C01 (+1).
        # Does it contribute to C10 or C11?
        # No, P5 is not in C10 or C11.
        u = [0]*10; u[0] = 1; u[5] = 1
        v = [0]*25; v[1*5 + k_offset] = 1; v[1*5 + k_offset+1] = 1
        w = [0]*10; w[k_offset*2 + 0] = -1; w[(k_offset+1)*2 + 0] = 1
        add_term(u, v, w)
        
        # Term 6: (A01 - A10 + A11) * (B00 + B01)
        # P6 = (A01 - A10 + A11)(B00 + B01)
        # Contributes to C10 (+1), C11 (+1).
        u = [0]*10; u[1] = 1; u[4] = -1; u[5] = 1
        v = [0]*25; v[0*5 + k_offset] = 1; v[0*5 + k_offset+1] = 1
        w = [0]*10; w[k_offset*2 + 1] = 1; w[(k_offset+1)*2 + 1] = 1
        add_term(u, v, w)
        
        # Term 7: (A00 + A01 - A10 - A11) * (B10 + B11)
        # P7 = (A00 + A01 - A10 - A11)(B10 + B11)
        # Contributes to C00 (+1).
        u = [0]*10; u[0] = 1; u[1] = 1; u[4] = -1; u[5] = -1
        v = [0]*25; v[1*5 + k_offset] = 1; v[1*5 + k_offset+1] = 1
        w = [0]*10; w[k_offset*2 + 0] = 1
        add_term(u, v, w)

    add_strassen_block(0)
    add_strassen_block(2)
    
    # B1 column 4: 2x1 block. Schoolbook.
    # C04 = A00 B04 + A01 B14
    # C14 = A10 B04 + A11 B14
    # Term 1: A00 B04
    u = [0]*10; u[0] = 1
    v = [0]*25; v[0*5 + 4] = 1
    w = [0]*10; w[4*2 + 0] = 1
    add_term(u, v, w)
    
    # Term 2: A01 B14
    u = [0]*10; u[1] = 1
    v = [0]*25; v[1*5 + 4] = 1
    w = [0]*10; w[4*2 + 0] = 1
    add_term(u, v, w)
    
    # Term 3: A10 B04
    u = [0]*10; u[4] = 1
    v = [0]*25; v[0*5 + 4] = 1
    w = [0]*10; w[4*2 + 1] = 1
    add_term(u, v, w)
    
    # Term 4: A11 B14
    u = [0]*10; u[5] = 1
    v = [0]*25; v[1*5 + 4] = 1
    w = [0]*10; w[4*2 + 1] = 1
    add_term(u, v, w)

    # Part 2: A2 (2x3) * B2 (3x5)
    # A2 indices: i in 0..1, j in 2..4 (global)
    # B2 indices: j in 2..4 (global), k in 0..4
    # Schoolbook: 2 * 3 * 5 = 30 terms.
    
    for i in range(2):
        for j in range(2, 5):
            for k in range(5):
                u = [0]*10; u[i*5 + j] = 1
                v = [0]*25; v[j*5 + k] = 1
                w = [0]*10; w[k*2 + i] = 1
                add_term(u, v, w)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
