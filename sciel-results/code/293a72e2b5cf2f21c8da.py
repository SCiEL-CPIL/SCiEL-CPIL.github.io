import numpy as np

DIMENSIONS = [2, 2, 5]

def construct():
    n, m, p = DIMENSIONS
    # U: n*m x R
    # V: m*p x R
    # W: p*n x R
    
    # We will build rank 18
    R = 18
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    # Indices for A (u)
    u00 = 0
    u01 = 1
    u10 = 2
    u11 = 3
    
    # Indices for B (v)
    # B is 2x5. v = j*p + k
    # j=0: k=0..4 -> 0..4
    # j=1: k=0..4 -> 5..9
    
    # Block 1: k=0,1
    b1_00 = 0
    b1_01 = 1
    b1_10 = 5
    b1_11 = 6
    
    # Block 2: k=2,3
    b2_00 = 2
    b2_01 = 3
    b2_10 = 7
    b2_11 = 8
    
    # Block 3: k=4
    b3_00 = 4
    b3_10 = 9
    
    # Indices for C (w)
    # C is 2x5. w = k*n + i
    # k=0: i=0,1 -> 0,1
    # k=1: i=0,1 -> 2,3
    # k=2: i=0,1 -> 4,5
    # k=3: i=0,1 -> 6,7
    # k=4: i=0,1 -> 8,9
    
    c1_00 = 0
    c1_01 = 2
    c1_10 = 1
    c1_11 = 3
    
    c2_00 = 4
    c2_01 = 6
    c2_10 = 5
    c2_11 = 7
    
    c3_00 = 8
    c3_10 = 9
    
    def add_strassen(base_r, b_idx, c_idx):
        # b_idx: dict '00','01','10','11' -> v_idx
        # c_idx: dict '00','01','10','11' -> w_idx
        
        # M1
        r = base_r
        U[u00, r] += 1
        U[u11, r] += 1
        V[b_idx['00'], r] += 1
        V[b_idx['11'], r] += 1
        W[c_idx['00'], r] += 1
        W[c_idx['01'], r] += 1
        W[c_idx['10'], r] += 1
        W[c_idx['11'], r] += 1
        
        # M2
        r = base_r + 1
        U[u10, r] += 1
        U[u11, r] += 1
        V[b_idx['00'], r] += 1
        W[c_idx['10'], r] += 1
        W[c_idx['11'], r] += 1
        
        # M3
        r = base_r + 2
        U[u00, r] += 1
        V[b_idx['01'], r] += 1
        V[b_idx['11'], r] -= 1
        W[c_idx['01'], r] += 1
        W[c_idx['11'], r] -= 1
        
        # M4
        r = base_r + 3
        U[u11, r] += 1
        V[b_idx['10'], r] += 1
        V[b_idx['00'], r] -= 1
        W[c_idx['00'], r] += 1
        W[c_idx['10'], r] += 1
        W[c_idx['11'], r] += 1
        
        # M5
        r = base_r + 4
        U[u00, r] += 1
        U[u01, r] += 1
        V[b_idx['11'], r] += 1
        W[c_idx['00'], r] -= 1
        W[c_idx['01'], r] += 1
        
        # M6
        r = base_r + 5
        U[u10, r] += 1
        U[u00, r] -= 1
        V[b_idx['00'], r] += 1
        V[b_idx['01'], r] += 1
        W[c_idx['01'], r] -= 1
        W[c_idx['10'], r] += 1
        
        # M7
        r = base_r + 6
        U[u01, r] += 1
        U[u11, r] -= 1
        V[b_idx['10'], r] += 1
        V[b_idx['11'], r] += 1
        W[c_idx['00'], r] += 1
        W[c_idx['10'], r] -= 1

    # Add Block 1
    add_strassen(0, 
                 {'00': b1_00, '01': b1_01, '10': b1_10, '11': b1_11},
                 {'00': c1_00, '01': c1_01, '10': c1_10, '11': c1_11})
                 
    # Add Block 2
    add_strassen(7,
                 {'00': b2_00, '01': b2_01, '10': b2_10, '11': b2_11},
                 {'00': c2_00, '01': c2_01, '10': c2_10, '11': c2_11})
                 
    # Add Block 3 (Schoolbook)
    # C04 = A00 B04 + A01 B14
    # C14 = A10 B04 + A11 B14
    
    r = 14
    U[u00, r] = 1
    V[b3_00, r] = 1
    W[c3_00, r] = 1
    
    r = 15
    U[u01, r] = 1
    V[b3_10, r] = 1
    W[c3_00, r] = 1
    
    r = 16
    U[u10, r] = 1
    V[b3_00, r] = 1
    W[c3_10, r] = 1
    
    r = 17
    U[u11, r] = 1
    V[b3_10, r] = 1
    W[c3_10, r] = 1
    
    return {"U": U, "V": V, "W": W}
