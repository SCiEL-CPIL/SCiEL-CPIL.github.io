def construct():
    n, m, p = 2, 4, 8
    # Use the 2x4x8 tensor decomposition based on 7 Strassen terms per 2x2x2 block
    # We have 4 k-blocks (each of width 2) and 2 j-blocks (each of width 2)
    # Total rank = 7 * 4 * 2 = 56
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    col = 0
    
    # Strassen terms for 2x2 matrix multiplication
    # Each term has U, V, W coefficients
    # U: list of (i, j, val) where i is row of A, j is col of A
    # V: list of (j, k, val) where j is row of B, k is col of B  
    # W: list of (i, k, val) where i is row of C, k is col of C
    
    strassen_terms = [
        # P1: (a+d)(e+h)
        {
            "U": [(0,0,1), (1,1,1)],
            "V": [(0,0,1), (1,1,1)],
            "W": [(0,0,1), (1,1,1)]
        },
        # P2: (c+d)e
        {
            "U": [(1,0,1), (1,1,1)],
            "V": [(0,0,1)],
            "W": [(1,0,1), (1,1,-1)]
        },
        # P3: a(f-h)
        {
            "U": [(0,0,1)],
            "V": [(0,1,1), (1,1,-1)],
            "W": [(0,1,1), (1,1,1)]
        },
        # P4: d(g-e)
        {
            "U": [(1,1,1)],
            "V": [(1,0,1), (0,0,-1)],
            "W": [(0,0,1), (1,0,1)]
        },
        # P5: (a+b)h
        {
            "U": [(0,0,1), (0,1,1)],
            "V": [(1,1,1)],
            "W": [(0,0,-1), (0,1,1)]
        },
        # P6: (c-a)(e+f)
        {
            "U": [(1,0,1), (0,0,-1)],
            "V": [(0,0,1), (0,1,1)],
            "W": [(1,1,1)]
        },
        # P7: (b-d)(g+h)
        {
            "U": [(0,1,1), (1,1,-1)],
            "V": [(1,0,1), (1,1,1)],
            "W": [(0,0,1)]
        },
    ]
    
    # Process each j-block (j in {0,1} and j in {2,3})
    for j_block in range(2):
        j_offset = 2 * j_block
        
        # Process each k-block (k in {0,1}, {2,3}, {4,5}, {6,7})
        for k_block in range(4):
            k_offset = 2 * k_block
            
            for term in strassen_terms:
                # Map local indices to global
                # A local (i, j_local) -> global (i, j_offset + j_local)
                # B local (j_local, k_local) -> global (j_offset + j_local, k_offset + k_local)
                # C local (i, k_local) -> global (i, k_offset + k_local)
                
                for (i, jl, val) in term["U"]:
                    jg = j_offset + jl
                    U[i * m + jg][col] += val
                
                for (jl, kl, val) in term["V"]:
                    jg = j_offset + jl
                    kg = k_offset + kl
                    V[jg * p + kg][col] += val
                
                for (i, kl, val) in term["W"]:
                    kg = k_offset + kl
                    W[kg * n + i][col] += val
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
