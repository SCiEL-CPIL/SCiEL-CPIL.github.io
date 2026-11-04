def construct():
    n, m, p = 4, 4, 7
    
    rank = 49 + 48
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 vectors
    # u, v, w are 2x2 matrices flattened to length 4
    # Index: row * 2 + col
    s_u = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    s_v = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 1, -1],
        [0, 1, -1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    s_w = [
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    col = 0
    # Part 1: 4x4x4 using Kronecker of Strassen
    for r1 in range(7):
        for r2 in range(7):
            # U, V, W for this rank step
            # Iterate over all 4x4 indices
            for I in range(2):
                for J in range(2):
                    for K in range(2):
                        for i in range(2):
                            for j in range(2):
                                for k in range(2):
                                    i_tot = 2 * I + i
                                    j_tot = 2 * J + j
                                    k_tot = 2 * K + k
                                    
                                    u_val = s_u[r1][I*2 + J] * s_u[r2][i*2 + j]
                                    v_val = s_v[r1][J*2 + K] * s_v[r2][j*2 + k]
                                    w_val = s_w[r1][K*2 + I] * s_w[r2][k*2 + i]
                                    
                                    u_idx = i_tot * 4 + j_tot
                                    v_idx = j_tot * 4 + k_tot
                                    w_idx = k_tot * 4 + i_tot
                                    
                                    U[u_idx][col] = u_val
                                    V[v_idx][col] = v_val
                                    W[w_idx][col] = w_val
            col += 1
            
    # Part 2: 4x4x3 schoolbook
    for i in range(4):
        for j in range(4):
            for k in range(3):
                u_idx = i * 4 + j
                v_idx = j * 4 + k  # Wait, p=7. V is 4x7. Index j*7 + k?
                # Prompt: V shape (m*p, rank). m=4, p=7.
                # So V index is j * 7 + k.
                # W shape (p*n, rank). p=7, n=4.
                # W index is k * 4 + i.
                
                # In Part 1, p was effectively 4.
                # In Part 2, p is 7.
                # The U, V, W matrices are sized for the full problem (4x4, 4x7, 7x4).
                # So indices must be consistent with full dimensions.
                
                U[u_idx][col] = 1
                V[j * 7 + k][col] = 1
                W[k * 4 + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
