def construct():
    n, m, p = 2, 2, 6
    rank = 21
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # U coefficients for A (2x2)
    # Indices: 0:00, 1:01, 2:10, 3:11
    U_coeffs = [
        {0: 1, 3: 1},       # M1: A00 + A11
        {2: 1, 3: 1},       # M2: A10 + A11
        {0: 1},             # M3: A00
        {3: 1},             # M4: A11
        {0: 1, 1: 1},       # M5: A00 + A01
        {2: 1, 0: -1},      # M6: A10 - A00
        {1: 1, 3: -1}       # M7: A01 - A11
    ]
    
    # V coefficients for B blocks (2x2 each, total 2x6)
    # V index: j * 6 + k
    # Block b uses cols 2b, 2b+1
    # (j, k_local, val)
    V_coeffs = [
        [(0, 0, 1), (1, 1, 1)],      # M1
        [(0, 0, 1)],                 # M2
        [(0, 1, 1), (1, 1, -1)],     # M3
        [(1, 0, 1), (0, 0, -1)],     # M4
        [(1, 1, 1)],                 # M5
        [(0, 0, 1), (0, 1, 1)],      # M6
        [(1, 0, 1), (1, 1, 1)]       # M7
    ]
    
    # W coefficients for C blocks (2x2 each, total 2x6)
    # W index: k * 2 + i
    # Block b uses cols 2b, 2b+1
    # (i, k_local, val)
    W_coeffs = [
        [(0, 0, 1), (1, 1, 1)],      # M1
        [(1, 0, 1), (1, 1, -1)],     # M2
        [(0, 1, 1), (1, 1, 1)],      # M3
        [(0, 0, 1), (1, 0, 1)],      # M4
        [(0, 0, -1), (0, 1, 1)],     # M5
        [(1, 1, 1)],                 # M6
        [(0, 0, 1)]                  # M7
    ]
    
    for b in range(3):
        for t in range(7):
            r = b * 7 + t
            
            # Fill U
            for idx, val in U_coeffs[t].items():
                U[idx][r] = val
            
            # Fill V
            for (j, k_local, val) in V_coeffs[t]:
                k_global = 2 * b + k_local
                v_idx = j * p + k_global
                V[v_idx][r] = val
            
            # Fill W
            for (i, k_local, val) in W_coeffs[t]:
                k_global = 2 * b + k_local
                w_idx = k_global * n + i
                W[w_idx][r] = val
                
    return {"U": U, "V": V, "W": W}
