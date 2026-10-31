def construct():
    n, m, p = 4, 4, 4
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication Z = X * Y
    # X, Y, Z are 2x2 matrices.
    # Indices 0,1 for rows/cols.
    # m_s = sum_{alpha,beta} u_s[alpha,beta] X[alpha,beta] * sum_{beta,gamma} v_s[beta,gamma] Y[beta,gamma]
    # Z[alpha,gamma] += w_s[alpha,gamma] * m_s
    
    # We define the 7 Strassen multiplications.
    # Each is a tuple: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict {(alpha, beta): coeff}
    # v_coeffs: dict {(beta, gamma): coeff}
    # w_coeffs: dict {(alpha, gamma): coeff}
    
    strassen_mults = [
        # m1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # m2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
        # m3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1}),
        # m4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # m5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # m6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # m7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
    ]
    
    # Map global indices to local indices
    # i = 2a + alpha
    # j = 2b + beta
    # k = 2c + gamma
    
    col = 0
    for a in range(2):
        for c in range(2):
            for b in range(2):
                # This corresponds to the product A[a,b] * B[b,c]
                # which contributes to C[a,c]
                
                for s in range(7):
                    u_coeffs, v_coeffs, w_coeffs = strassen_mults[s]
                    
                    # Populate U
                    for (alpha, beta), coeff in u_coeffs.items():
                        i = 2 * a + alpha
                        j = 2 * b + beta
                        U[i * m + j][col] += coeff
                    
                    # Populate V
                    for (beta, gamma), coeff in v_coeffs.items():
                        j = 2 * b + beta
                        k = 2 * c + gamma
                        V[j * p + k][col] += coeff
                    
                    # Populate W
                    for (alpha, gamma), coeff in w_coeffs.items():
                        i = 2 * a + alpha
                        k = 2 * c + gamma
                        # W index is k * n + i
                        W[k * n + i][col] += coeff
                    
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
