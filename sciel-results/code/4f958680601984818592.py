import numpy as np

def construct():
    n, m, p = 2, 2, 13
    # Rank 46 decomposition
    # 6 blocks of 2 columns (Strassen rank 7) + 1 block of 1 column (Rank 4)
    rank = 6 * 7 + 4
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col_offset = 0
    
    # Strassen coefficients for 2x2
    # A = [[a0, a1], [a2, a3]]
    # B = [[b0, b1], [b2, b3]]
    # C = [[c0, c1], [c2, c3]]
    #
    # m1 = (a0 + a3)(b0 + b3)
    # m2 = (a2 + a3)b0
    # m3 = (a0 + a1)b3
    # m4 = (a0 + a2)(b2 + b0)
    # m5 = (a0 + a1)(b3 - b2)
    # m6 = (a1 - a3)(b3 + b2)
    # m7 = (a1 - a2)(b0 + b1)
    #
    # c0 = m1 + m4 - m5 - m7
    # c1 = m5 + m6
    # c2 = m3 + m4
    # c3 = m1 - m2 + m4 + m7
    
    # U coeffs for A
    # m1: a0=1, a3=1
    # m2: a2=1, a3=1
    # m3: a0=1, a1=1
    # m4: a0=1, a2=1
    # m5: a0=1, a1=1
    # m6: a1=1, a3=-1
    # m7: a1=1, a2=-1
    
    u_strassen = [
        [1, 0, 1, 0], # m1
        [0, 0, 1, 1], # m2
        [1, 1, 0, 0], # m3
        [1, 0, 1, 0], # m4
        [1, 1, 0, 0], # m5
        [0, 1, 0, -1],# m6
        [0, 1, -1, 0] # m7
    ]
    
    # V coeffs for B (local to block)
    # m1: b0=1, b3=1
    # m2: b0=1
    # m3: b3=1
    # m4: b2=1, b0=1
    # m5: b3=1, b2=-1
    # m6: b3=1, b2=1
    # m7: b0=1, b1=1
    
    v_strassen = [
        [1, 0, 0, 1], # m1
        [1, 0, 0, 0], # m2
        [0, 0, 0, 1], # m3
        [1, 0, 1, 0], # m4
        [0, 0, -1, 1],# m5
        [0, 0, 1, 1], # m6
        [1, 1, 0, 0]  # m7
    ]
    
    # W coeffs for C (local to block)
    # c0: m1=1, m4=1, m5=-1, m7=-1
    # c1: m5=1, m6=1
    # c2: m3=1, m4=1
    # c3: m1=1, m2=-1, m4=1, m7=1
    
    w_strassen = [
        [1, 0, 0, 1, -1, 0, -1], # c0
        [0, 0, 0, 0, 1, 1, 0],   # c1
        [0, 0, 1, 1, 0, 0, 0],   # c2
        [1, -1, 0, 1, 0, 0, 1]   # c3
    ]
    
    for k in range(6):
        # Block k uses B columns 2k, 2k+1
        # V indices: 0*13+2k, 0*13+2k+1, 1*13+2k, 1*13+2k+1
        v_indices = [13*0 + 2*k, 13*0 + 2*k+1, 13*1 + 2*k, 13*1 + 2*k+1]
        
        # W indices: C columns 2k, 2k+1
        # W index formula: col * n + row = col * 2 + row
        # c0 (row 0, col 2k) -> 2k*2 + 0
        # c1 (row 0, col 2k+1) -> (2k+1)*2 + 0
        # c2 (row 1, col 2k) -> 2k*2 + 1
        # c3 (row 1, col 2k+1) -> (2k+1)*2 + 1
        w_indices = [2*k*2 + 0, (2*k+1)*2 + 0, 2*k*2 + 1, (2*k+1)*2 + 1]
        
        for r in range(7):
            U[:, col_offset] = u_strassen[r]
            for i, idx in enumerate(v_indices):
                V[idx, col_offset] = v_strassen[r][i]
            for i, idx in enumerate(w_indices):
                W[idx, col_offset] = w_strassen[r][i]
            col_offset += 1
            
    # Last block: 1 column (col 12)
    # Rank 4 decomposition
    # c0 = a0 b0 + a1 b1
    # c1 = a2 b0 + a3 b1
    
    u_simple = [
        [1, 0, 0, 0], # a0
        [0, 1, 0, 0], # a1
        [0, 0, 1, 0], # a2
        [0, 0, 0, 1]  # a3
    ]
    
    v_simple = [
        [1, 0], # b0
        [0, 1], # b1
        [1, 0], # b0
        [0, 1]  # b1
    ]
    
    w_simple = [
        [1, 0, 0, 0], # c0
        [0, 1, 0, 0], # c1
        [0, 0, 1, 0], # c1 (row 1)
        [0, 0, 0, 1]  # c1 (row 1) -- Wait
    ]
    # c0 (row 0) = a0 b0 + a1 b1
    # c1 (row 1) = a2 b0 + a3 b1
    
    w_simple = [
        [1, 1, 0, 0], # c0: m1(a0b0) + m2(a1b1)
        [0, 0, 1, 1]  # c1: m3(a2b0) + m4(a3b1)
    ]
    # Wait, W has 4 rows for this block?
    # W indices for col 12:
    # c0 (row 0, col 12) -> 12*2 + 0
    # c1 (row 1, col 12) -> 12*2 + 1
    
    v_indices_simple = [13*0 + 12, 13*1 + 12]
    w_indices_simple = [12*2 + 0, 12*2 + 1]
    
    for r in range(4):
        U[:, col_offset] = u_simple[r]
        for i, idx in enumerate(v_indices_simple):
            V[idx, col_offset] = v_simple[r][i]
        for i, idx in enumerate(w_indices_simple):
            W[idx, col_offset] = w_simple[r][i]
        col_offset += 1

    return {"U": U, "V": V, "W": W}
