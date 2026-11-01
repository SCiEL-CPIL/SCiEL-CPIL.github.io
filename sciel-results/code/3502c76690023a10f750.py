def construct():
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # Indices:
    # A: 0,0 0,1 1,0 1,1 -> U indices 0,1,2,3
    # B: 0,0 0,1 1,0 1,1 -> V indices 0,1,2,3 (for k=0,1)
    # C: 0,0 0,1 1,0 1,1 -> W indices 0,1,2,3 (for k=0,1)
    
    # Define the 7 Strassen terms
    # Each term is (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs is dict {idx: val}
    # V_coeffs is dict {idx: val}
    # W_coeffs is dict {idx: val}
    
    strassen_terms = [
        # M1 = (a+d)(e+h)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (c+d)e
        ({2: 1, 3: 1}, {0: 1}, {1: 1}),
        # M3 = a(f-h)
        ({0: 1}, {1: 1, 3: -1}, {2: 1, 3: 1}),
        # M4 = d(g-e)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 1: 1}),
        # M5 = (a+b)h
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 2: 1}),
        # M6 = (c-a)(e+f)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (b-d)(g+h)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]
    
    # Fill for first block (k=0,1)
    for r in range(7):
        u_dict, v_dict, w_dict = strassen_terms[r]
        for idx, val in u_dict.items():
            U[idx][r] = val
        for idx, val in v_dict.items():
            # V index is j*p + k. Here p=4.
            # But for the first block, we are using local indices 0,1 for j,k.
            # The global V index is j*4 + k.
            # The local indices in strassen_terms for V are 0,1,2,3 corresponding to (0,0), (0,1), (1,0), (1,1).
            # So local_j, local_k = divmod(idx, 2)? No, idx is 0..3.
            # 0 -> (0,0), 1 -> (0,1), 2 -> (1,0), 3 -> (1,1).
            # So local_j = idx // 2, local_k = idx % 2.
            local_j = idx // 2
            local_k = idx % 2
            global_v_idx = local_j * p + local_k
            V[global_v_idx][r] = val
        for idx, val in w_dict.items():
            # W index is k*n + i.
            # Local indices 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1) for (i,k)?
            # My W dict keys were:
            # 0: C00 (i=0, k=0)
            # 1: C10 (i=1, k=0) -- Wait, in M2 I used W[1] for C10.
            # 2: C01 (i=0, k=1)
            # 3: C11 (i=1, k=1)
            # So local_i = idx % 2?
            # 0 -> i=0, k=0.
            # 1 -> i=1, k=0.
            # 2 -> i=0, k=1.
            # 3 -> i=1, k=1.
            # So local_i = idx % 2, local_k = idx // 2.
            local_i = idx % 2
            local_k = idx // 2
            global_w_idx = local_k * n + local_i
            W[global_w_idx][r] = val

    # Fill for second block (k=2,3)
    # The U coefficients are the same.
    # The V coefficients are the same, but mapped to k=2,3.
    # The W coefficients are the same, but mapped to k=2,3.
    
    for r in range(7):
        u_dict, v_dict, w_dict = strassen_terms[r]
        for idx, val in u_dict.items():
            U[idx][r + 7] = val
        for idx, val in v_dict.items():
            local_j = idx // 2
            local_k = idx % 2
            # Map local_k to global k: 0->2, 1->3
            global_k = local_k + 2
            global_v_idx = local_j * p + global_k
            V[global_v_idx][r + 7] = val
        for idx, val in w_dict.items():
            local_i = idx % 2
            local_k = idx // 2
            global_k = local_k + 2
            global_w_idx = global_k * n + local_i
            W[global_w_idx][r + 7] = val

    return {"U": U, "V": V, "W": W}
