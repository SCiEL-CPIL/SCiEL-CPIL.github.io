def construct():
    n, m, p = DIMENSIONS
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec, r):
        for i in range(len(u_vec)):
            if u_vec[i] != 0:
                U[i][r] = u_vec[i]
        for i in range(len(v_vec)):
            if v_vec[i] != 0:
                V[i][r] = v_vec[i]
        for i in range(len(w_vec)):
            if w_vec[i] != 0:
                W[i][r] = w_vec[i]

    # Strassen coefficients for 2x2x2
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    strassen_terms = [
        # (U, V, W)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # P1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # P2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # P3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]), # P4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # P5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # P6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])  # P7
    ]
    
    # Block 1: Cols 0, 1 (V indices 0-3, W indices 0-3)
    for i, (u, v, w) in enumerate(strassen_terms):
        r = i
        add_term(u, v, w, r)
        
    # Block 2: Cols 2, 3 (V indices 4-7, W indices 4-7)
    for i, (u, v, w) in enumerate(strassen_terms):
        r = 7 + i
        v_shifted = [0]*10
        w_shifted = [0]*10
        for j in range(4):
            v_shifted[4+j] = v[j]
            w_shifted[4+j] = w[j]
        add_term(u, v_shifted, w_shifted, r)
        
    # Block 3: Col 4 (V indices 8-9, W indices 8-9)
    # Schoolbook
    # A00 B04 -> C04
    add_term([1, 0, 0, 0], [0,0,0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,0,1,0], 14)
    # A01 B14 -> C04
    add_term([0, 1, 0, 0], [0,0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1,0], 15)
    # A10 B04 -> C14
    add_term([0, 0, 1, 0], [0,0,0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,0,0,1], 16)
    # A11 B14 -> C14
    add_term([0, 0, 0, 1], [0,0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,0,1], 17)
    
    return {"U": U, "V": V, "W": W}
