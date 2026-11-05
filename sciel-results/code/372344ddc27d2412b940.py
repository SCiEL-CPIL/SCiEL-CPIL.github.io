def construct():
    n, m, p = 2, 4, 8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2 multiplication
    # A indices: 00, 01, 10, 11
    # B indices: 00, 01, 10, 11
    # C indices: 00, 01, 10, 11
    
    # Terms defined as (U_coefs, V_coefs, W_coefs)
    # Coefs are (row, col, value)
    
    strassen_terms = [
        # P1: (a+d)(e+h) -> C00+, C11+
        {
            "U": [(0,0,1), (1,1,1)],
            "V": [(0,0,1), (1,1,1)],
            "W": [(0,0,1), (1,1,1)]
        },
        # P2: (c+d)e    -> C10+, C11-
        {
            "U": [(1,0,1), (1,1,1)],
            "V": [(0,0,1)],
            "W": [(1,0,1), (1,1,-1)]
        },
        # P3: a(f-h)    -> C01+, C11+
        {
            "U": [(0,0,1)],
            "V": [(0,1,1), (1,1,-1)],
            "W": [(0,1,1), (1,1,1)]
        },
        # P4: d(g-e)    -> C00+, C10+
        {
            "U": [(1,1,1)],
            "V": [(1,0,1), (0,0,-1)],
            "W": [(0,0,1), (1,0,1)]
        },
        # P5: (a+b)h    -> C00-, C01+
        {
            "U": [(0,0,1), (0,1,1)],
            "V": [(1,1,1)],
            "W": [(0,0,-1), (0,1,1)]
        },
        # P6: (c-a)(e+f)-> C11+
        {
            "U": [(1,0,1), (0,0,-1)],
            "V": [(0,0,1), (0,1,1)],
            "W": [(1,1,1)]
        },
        # P7: (b-d)(g+h)-> C00+
        {
            "U": [(0,1,1), (1,1,-1)],
            "V": [(1,0,1), (1,1,1)],
            "W": [(0,0,1)]
        },
    ]
    
    # Process j-blocks {0,1} and {2,3}
    # Process k-blocks {0,1}, {2,3}, {4,5}, {6,7}
    
    for j_offset in [0, 2]:
        for k_offset in [0, 2, 4, 6]:
            for term in strassen_terms:
                # U: A[i, j]
                for (i, jl, val) in term["U"]:
                    jg = j_offset + jl
                    U[i*4 + jg][col] += val
                
                # V: B[j, k]
                for (jl, kl, val) in term["V"]:
                    jg = j_offset + jl
                    kg = k_offset + kl
                    V[jg*8 + kg][col] += val
                
                # W: C[i, k] -> W[k*2 + i]
                for (i, kl, val) in term["W"]:
                    kg = k_offset + kl
                    W[kg*2 + i][col] += val
                
                col += 1
            
    return {"U": U, "V": V, "W": W}
