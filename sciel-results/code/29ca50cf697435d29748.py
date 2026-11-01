def construct():
    n, m, p = 2, 4, 8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms: (U_loc, V_loc, W_loc)
    # Indices: 0:00, 1:01, 2:10, 3:11
    terms = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # M4
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    col_idx = 0
    for k_group in range(4):
        for j_group in range(2):
            for u_loc, v_loc, w_loc in terms:
                # Map U
                for l in range(4):
                    if u_loc[l] != 0:
                        i = l // 2
                        j = l % 2
                        if j_group == 1:
                            j += 2
                        u_idx = i * m + j
                        U[u_idx][col_idx] += u_loc[l]
                
                # Map V
                for l in range(4):
                    if v_loc[l] != 0:
                        j = l // 2
                        k = l % 2
                        if j_group == 1:
                            j += 2
                        v_idx = j * p + (k_group * 2 + k)
                        V[v_idx][col_idx] += v_loc[l]
                        
                # Map W
                for l in range(4):
                    if w_loc[l] != 0:
                        k = l // 2
                        i = l % 2
                        w_idx = (k_group * 2 + k) * n + i
                        W[w_idx][col_idx] += w_loc[l]
                        
                col_idx += 1
                
    return {"U": U, "V": V, "W": W}
