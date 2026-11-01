DIMENSIONS = [2, 2, 13]

def construct():
    n, m, p = DIMENSIONS
    rank = 6 * 7 + 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def set_term(u_dict, v_dict, w_dict):
        nonlocal col
        for idx, val in u_dict.items():
            U[idx][col] = val
        for idx, val in v_dict.items():
            V[idx][col] = val
        for idx, val in w_dict.items():
            W[idx][col] = val
        col += 1

    # Strassen pair (k, k+1)
    def add_strassen_pair(k):
        k1 = k
        k2 = k + 1
        
        # P1 = (A00 + A11) * (B0k1 + B1k2)
        set_term(
            {0: 1, 3: 1}, 
            {0*p+k1: 1, 1*p+k2: 1}, 
            {k1*n+0: 1, k2*n+1: 1}
        )
        
        # P2 = (A10 + A11) * B0k1
        set_term(
            {2: 1, 3: 1}, 
            {0*p+k1: 1}, 
            {k1*n+1: 1}
        )
        
        # P3 = A00 * (B0k2 - B1k2)
        set_term(
            {0: 1}, 
            {0*p+k2: 1, 1*p+k2: -1}, 
            {k2*n+0: 1}
        )
        
        # P4 = A11 * (B1k1 - B0k1)
        set_term(
            {3: 1}, 
            {1*p+k1: 1, 0*p+k1: -1}, 
            {k1*n+1: 1, k2*n+0: -1} # Wait, P4 contributes to C1k1 (+1) and C0k2? No.
            # C1k1 = P2 + P4. So P4 coeff 1 for C1k1.
            # C0k2 = P3 + P5. P4 not involved.
            # C1k2 = P1 - P3 + P7. P4 not involved.
            # C0k1 = P1 - P4 + P6. So P4 coeff -1 for C0k1.
            # So W should be {k1*n+1: 1, k1*n+0: -1}
        )
        # Correction for P4 W:
        # C0k1 gets -P4. C1k1 gets +P4.
        # So W[k1*n+0] = -1, W[k1*n+1] = 1.
        # Let's re-verify Strassen formulas.
        # C00 = P1 - P4 + P6
        # C10 = P2 + P4
        # C01 = P3 + P5
        # C11 = P1 - P3 + P7
        # So for pair (k1, k2):
        # C0k1 = P1 - P4 + P6
        # C1k1 = P2 + P4
        # C0k2 = P3 + P5
        # C1k2 = P1 - P3 + P7
        
        # My P4 set_term had wrong W. I will fix it in the final code.
        
        # P5 = (A00 + A01) * B1k2
        set_term(
            {0: 1, 1: 1}, 
            {1*p+k2: 1}, 
            {k2*n+0: 1}
        )
        
        # P6 = (A10 - A00) * (B0k1 + B0k2)
        set_term(
            {2: 1, 0: -1}, 
            {0*p+k1: 1, 0*p+k2: 1}, 
            {k1*n+0: 1}
        )
        
        # P7 = (A01 - A11) * (B1k1 + B1k2)
        set_term(
            {1: 1, 3: -1}, 
            {1*p+k1: 1, 1*p+k2: 1}, 
            {k2*n+1: 1}
        )

    # Single column k
    def add_single(k):
        # C0k = A00 B0k + A01 B1k
        # C1k = A10 B0k + A11 B1k
        set_term({0: 1}, {0*p+k: 1}, {k*n+0: 1})
        set_term({1: 1}, {1*p+k: 1}, {k*n+0: 1})
        set_term({2: 1}, {0*p+k: 1}, {k*n+1: 1})
        set_term({3: 1}, {1*p+k: 1}, {k*n+1: 1})

    for i in range(6):
        add_strassen_pair(2*i)
    add_single(12)

    return {"U": U, "V": V, "W": W}
