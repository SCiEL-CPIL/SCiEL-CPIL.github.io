def construct():
    n, m, p = 2, 4, 8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients
    # P1: (a+d)(e+h) -> C00+, C11+
    # P2: (c+d)e    -> C10+, C11-
    # P3: a(f-h)    -> C01+, C11+
    # P4: d(g-e)    -> C00+, C10+
    # P5: (a+b)h    -> C00-, C01+
    # P6: (c-a)(e+f)-> C11+
    # P7: (b-d)(g+h)-> C00+
    
    # We define the terms for one 2x2 block multiplication
    # A indices: 00, 01, 10, 11
    # B indices: 00, 01, 10, 11
    # C indices: 00, 01, 10, 11
    
    # Term 1: P1
    # U: a, d
    # V: e, h
    # W: C00, C11
    terms = [
        # (U_coefs, V_coefs, W_coefs)
        # P1
        ([(0,1), (1,1)], [(0,1), (1,1)], [(0,0), (1,1)]),
        # P2
        ([(1,0), (1,1)], [(0,0)], [(1,0), (1,1)]),
        # P3
        ([(0,0)], [(0,1), (1,1)], [(0,1), (1,1)]),
        # P4
        ([(1,1)], [(1,0), (0,0)], [(0,0), (1,0)]),
        # P5
        ([(0,0), (0,1)], [(1,1)], [(0,0), (0,1)]),
        # P6
        ([(1,0), (0,0)], [(0,0), (0,1)], [(1,1)]),
        # P7
        ([(0,1), (1,1)], [(1,0), (1,1)], [(0,0)]),
    ]
    
    # Signs for W
    # C00: P1+ P4- P5- P7+
    # C01: P3+ P5+
    # C10: P2+ P4+
    # C11: P1- P2- P3+ P6+
    
    w_signs = [
        [1, 0, 0, 1], # P1, P2, P3, P4 ... wait, need to map correctly
        # Let's just list the contribution of each P to each C
        # P1: C00(1), C11(1)
        # P2: C10(1), C11(-1)
        # P3: C01(1), C11(1)
        # P4: C00(1), C10(1)
        # P5: C00(-1), C01(1)
        # P6: C11(1)
        # P7: C00(1)
        
        # Corrected signs:
        # P1: C00: 1, C11: 1
        # P2: C10: 1, C11: -1
        # P3: C01: 1, C11: 1
        # P4: C00: 1, C10: 1
        # P5: C00: -1, C01: 1
        # P6: C11: 1
        # P7: C00: 1
    ]
    
    # Re-defining terms with explicit signs in W
    # U_coefs: list of (i, j, val)
    # V_coefs: list of (j, k, val)
    # W_coefs: list of (i, k, val)
    
    strassen_terms = [
        # P1
        {
            "U": [(0,0,1), (1,1,1)],
            "V": [(0,0,1), (1,1,1)],
            "W": [(0,0,1), (1,1,1)]
        },
        # P2
        {
            "U": [(1,0,1), (1,1,1)],
            "V": [(0,0,1)],
            "W": [(1,0,1), (1,1,-1)]
        },
        # P3
        {
            "U": [(0,0,1)],
            "V": [(0,1,1), (1,1,-1)],
            "W": [(0,1,1), (1,1,1)]
        },
        # P4
        {
            "U": [(1,1,1)],
            "V": [(1,0,1), (0,0,-1)],
            "W": [(0,0,1), (1,0,1)]
        },
        # P5
        {
            "U": [(0,0,1), (0,1,1)],
            "V": [(1,1,1)],
            "W": [(0,0,-1), (0,1,1)]
        },
        # P6
        {
            "U": [(1,0,1), (0,0,-1)],
            "V": [(0,0,1), (0,1,1)],
            "W": [(1,1,1)]
        },
        # P7
        {
            "U": [(0,1,1), (1,1,-1)],
            "V": [(1,0,1), (1,1,1)],
            "W": [(0,0,1)]
        },
    ]
    
    # Process Set 1: j in {0,1}, k_blocks in {0,1,2,3}
    for k_block in range(4):
        for term in strassen_terms:
            # Map local indices to global
            # A local (i, j_local) -> global (i, j_global)
            # j_local 0->0, 1->1
            # B local (j_local, k_local) -> global (j_global, k_global)
            # j_local 0->0, 1->1
            # k_local 0->2*k_block, 1->2*k_block+1
            # C local (i, k_local) -> global (i, k_global)
            
            for (i, jl, val) in term["U"]:
                jg = jl
                U[i*4 + jg][col] += val
            
            for (jl, kl, val) in term["V"]:
                jg = jl
                kg = 2*k_block + kl
                V[jg*8 + kg][col] += val
            
            for (i, kl, val) in term["W"]:
                kg = 2*k_block + kl
                W[kg*2 + i][col] += val
            
            col += 1
            
    # Process Set 2: j in {2,3}, k_blocks in {0,1,2,3}
    for k_block in range(4):
        for term in strassen_terms:
            # Map local indices to global
            # A local (i, j_local) -> global (i, j_global)
            # j_local 0->2, 1->3
            # B local (j_local, k_local) -> global (j_global, k_global)
            # j_local 0->2, 1->3
            # k_local 0->2*k_block, 1->2*k_block+1
            
            for (i, jl, val) in term["U"]:
                jg = 2 + jl
                U[i*4 + jg][col] += val
            
            for (jl, kl, val) in term["V"]:
                jg = 2 + jl
                kg = 2*k_block + kl
                V[jg*8 + kg][col] += val
            
            for (i, kl, val) in term["W"]:
                kg = 2*k_block + kl
                W[kg*2 + i][col] += val
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
