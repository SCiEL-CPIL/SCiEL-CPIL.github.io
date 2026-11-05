def construct():
    n, m, p = 2, 4, 8
    rank = 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen terms for 2x2 block multiplication
    # Each term: (U_entries, V_entries, W_entries)
    # U_entries: list of (i, j, val) where i,j are local 0,1
    # V_entries: list of (j, k, val) where j,k are local 0,1
    # W_entries: list of (i, k, val) where i,k are local 0,1
    
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
    
    # Process all 2x2 blocks: i in {0,1}, j_blocks in {0,1}, k_blocks in {0,1,2,3}
    # A is 2x4, B is 4x8, C is 2x8
    # We partition B into 2x2 blocks along the j dimension (4 = 2*2) and k dimension (8 = 4*2)
    # Actually, let's think of it as: for each pair of rows of B (j=0,1 and j=2,3)
    # and each pair of columns of B (k=0,1; k=2,3; k=4,5; k=6,7)
    # we do a 2x2 multiplication
    
    for j_block in range(2):  # j_block 0 -> j={0,1}, j_block 1 -> j={2,3}
        for k_block in range(4):  # k_block 0 -> k={0,1}, etc.
            for term in strassen_terms:
                for (i, jl, val) in term["U"]:
                    jg = j_block * 2 + jl
                    U[i * 4 + jg][col] += val
                
                for (jl, kl, val) in term["V"]:
                    jg = j_block * 2 + jl
                    kg = k_block * 2 + kl
                    V[jg * 8 + kg][col] += val
                
                for (i, kl, val) in term["W"]:
                    kg = k_block * 2 + kl
                    W[kg * 2 + i][col] += val
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
