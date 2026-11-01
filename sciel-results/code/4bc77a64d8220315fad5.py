# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Shapes
    u_dim = n * m  # 4
    v_dim = m * p  # 8
    w_dim = p * n  # 8
    
    rank = 14
    U = [[0] * rank for _ in range(u_dim)]
    V = [[0] * rank for _ in range(v_dim)]
    W = [[0] * rank for _ in range(w_dim)]
    
    # Define the 7 Strassen terms for 2x2x2
    # Each term is (u_dict, v_dict, w_dict)
    # u_dict: {(i,j): coeff}
    # v_dict: {(j,k): coeff}
    # w_dict: {(k,i): coeff}
    
    terms = [
        # M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1, (1,1): -1}),
        # M3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
        # M4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
        # M5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(1,0): 1, (0,0): -1}),
        # M6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    # Fill for Block 1 (k in 0..1)
    for r in range(7):
        u_d, v_d, w_d = terms[r]
        col = r
        for (i,j), val in u_d.items():
            U[i*m + j][col] += val
        for (j,k), val in v_d.items():
            V[j*p + k][col] += val
        for (k,i), val in w_d.items():
            W[k*n + i][col] += val
            
    # Fill for Block 2 (k in 2..3)
    # The multiplication is C = A * B.
    # B is 2x4. We split B into B1 (cols 0,1) and B2 (cols 2,3).
    # C1 = A * B1, C2 = A * B2.
    # The Strassen terms for C2 use the same A (U) but shifted B (V) and shifted C (W).
    for r in range(7):
        u_d, v_d, w_d = terms[r]
        col = 7 + r
        for (i,j), val in u_d.items():
            U[i*m + j][col] += val
        for (j,k), val in v_d.items():
            k_new = k + 2
            V[j*p + k_new][col] += val
        for (k,i), val in w_d.items():
            k_new = k + 2
            W[k_new*n + i][col] += val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
