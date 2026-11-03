import numpy as np

def construct():
    n, m, p = 2, 4, 7
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Rank 50 decomposition
    R = 50
    U = np.zeros((nm, R))
    V = np.zeros((mp, R))
    W = np.zeros((pn, R))
    
    col = 0
    
    # Define Strassen terms for 2x2x2
    # Each term is a list of (i, j, val) for U, (j, k, val) for V, (k, i, val) for W
    # Indices are local 0,1
    strassen_terms = [
        # M1: (a00+a11)(b00+b11) -> c00(+), c11(+)
        [ [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)] ],
        # M2: (a10+a11)b00 -> c10(+)
        [ [(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1)] ],
        # M3: a00(b01-b11) -> c01(+), c11(+)
        [ [(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)] ],
        # M4: a11(b10-b00) -> c00(+), c10(+)
        [ [(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)] ],
        # M5: (a00+a10)b11 -> c00(-), c01(+)
        [ [(0,0,1), (1,0,1)], [(1,1,1)], [(0,0,-1), (0,1,1)] ],
        # M6: (a01-a00)(b00+b10) -> c00(-)
        [ [(0,1,1), (0,0,-1)], [(0,0,1), (1,0,1)], [(0,0,-1)] ],
        # M7: (a11-a10)(b01+b11) -> c00(-), c11(+)
        [ [(1,1,1), (1,0,-1)], [(0,1,1), (1,1,1)], [(0,0,-1), (1,1,1)] ],
    ]
    
    # Define Standard terms for 2x2x1
    # c00 = a00b00 + a01b10
    # c10 = a10b00 + a11b10
    standard_terms = [
        # a00 b00 -> c00
        [ [(0,0,1)], [(0,0,1)], [(0,0,1)] ],
        # a01 b10 -> c00
        [ [(0,1,1)], [(1,0,1)], [(0,0,1)] ],
        # a10 b00 -> c10
        [ [(1,0,1)], [(0,0,1)], [(1,0,1)] ],
        # a11 b10 -> c10
        [ [(1,1,1)], [(1,0,1)], [(1,0,1)] ],
    ]
    
    # Process 3 blocks of width 2
    for b in range(3):
        k_base = 2 * b
        
        # Part 1: A_L B_T
        for term in strassen_terms:
            u_entries, v_entries, w_entries = term
            for i, j, val in u_entries:
                U[i * 4 + j, col] += val
            for j, k, val in v_entries:
                V[j * 7 + (k_base + k), col] += val
            for k, i, val in w_entries:
                W[(k_base + k) * 2 + i, col] += val
            col += 1
        
        # Part 2: A_R B_B
        for term in strassen_terms:
            u_entries, v_entries, w_entries = term
            for i, j, val in u_entries:
                U[i * 4 + (2 + j), col] += val
            for j, k, val in v_entries:
                V[(2 + j) * 7 + (k_base + k), col] += val
            for k, i, val in w_entries:
                W[(k_base + k) * 2 + i, col] += val
            col += 1

    # Process 1 block of width 1 (k=6)
    k_base = 6
    
    # Part 1: A_L B_T
    for term in standard_terms:
        u_entries, v_entries, w_entries = term
        for i, j, val in u_entries:
            U[i * 4 + j, col] += val
        for j, k, val in v_entries:
            V[j * 7 + k_base, col] += val
        for k, i, val in w_entries:
            W[k_base * 2 + i, col] += val
        col += 1
        
    # Part 2: A_R B_B
    for term in standard_terms:
        u_entries, v_entries, w_entries = term
        for i, j, val in u_entries:
            U[i * 4 + (2 + j), col] += val
        for j, k, val in v_entries:
            V[(2 + j) * 7 + k_base, col] += val
        for k, i, val in w_entries:
            W[k_base * 2 + i, col] += val
        col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
