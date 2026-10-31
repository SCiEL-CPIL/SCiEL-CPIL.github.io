import time

def construct():
    start_time = time.time()
    deadline = start_time + 10.0
    
    n, m, p = 4, 4, 4
    # Recursive Strassen rank for 4x4 is 7*7 = 49
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # beta[s][I][J] for A[I][J]
    beta = [
        (1, 0, 0, 1),  # s=0: A00 + A11
        (0, 0, 1, 1),  # s=1: A10 + A11
        (1, 0, 0, 0),  # s=2: A00
        (0, 0, 0, 1),  # s=3: A11
        (1, 1, 0, 0),  # s=4: A00 + A01
        (-1, 0, 1, 0), # s=5: A10 - A00
        (0, 1, 0, -1)  # s=6: A01 - A11
    ]
    
    # gamma[s][J][K] for B[J][K]
    gamma = [
        (1, 0, 0, 1),  # s=0: B00 + B11
        (1, 0, 0, 0),  # s=1: B00
        (0, 1, 0, -1), # s=2: B01 - B11
        (-1, 0, 1, 0), # s=3: B10 - B00
        (0, 0, 0, 1),  # s=4: B11
        (1, 1, 0, 0),  # s=5: B00 + B01
        (0, 0, 1, 1)   # s=6: B10 + B11
    ]
    
    # delta[s][I][K] for C[I][K]
    delta = [
        (1, 0, 0, 1),  # s=0: C00 + C11
        (0, 0, 1, -1), # s=1: C10 - C11
        (0, 1, 0, 1),  # s=2: C01 + C11
        (1, 0, 1, 0),  # s=3: C00 + C10
        (-1, 1, 0, 0), # s=4: -C00 + C01
        (0, 0, 0, 1),  # s=5: C11
        (1, 0, 0, 0)   # s=6: C00
    ]

    # Helper to get coefficient from flattened tuple
    def get_coeff(coeffs, r, c):
        return coeffs[r * 2 + c]

    # Construct the 49-rank decomposition
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            
            # U[i*4+j, col] = beta[s][I][J] * beta[t][i'][j']
            # i = I*2 + i', j = J*2 + j'
            # I, J are outer indices (0,1), i', j' are inner indices (0,1)
            for I in range(2):
                for J in range(2):
                    bs = get_coeff(beta[s], I, J)
                    if bs == 0: continue
                    for ip in range(2):
                        for jp in range(2):
                            bt = get_coeff(beta[t], ip, jp)
                            if bt == 0: continue
                            row_u = (I * 2 + ip) * 4 + (J * 2 + jp)
                            U[row_u][col] += bs * bt

            # V[j*4+k, col] = gamma[s][J][K] * gamma[t][j'][k']
            for J in range(2):
                for K in range(2):
                    gs = get_coeff(gamma[s], J, K)
                    if gs == 0: continue
                    for jp in range(2):
                        for kp in range(2):
                            gt = get_coeff(gamma[t], jp, kp)
                            if gt == 0: continue
                            row_v = (J * 2 + jp) * 4 + (K * 2 + kp)
                            V[row_v][col] += gs * gt

            # W[k*4+i, col] = delta[s][I][K] * delta[t][i'][k']
            # Convention: W[k*n+i, r] where n=4. So row = k*4 + i.
            for I in range(2):
                for K in range(2):
                    ds = get_coeff(delta[s], I, K)
                    if ds == 0: continue
                    for ip in range(2):
                        for kp in range(2):
                            dt = get_coeff(delta[t], ip, kp)
                            if dt == 0: continue
                            # i = I*2 + ip, k = K*2 + kp
                            row_w = (K * 2 + kp) * 4 + (I * 2 + ip)
                            W[row_w][col] += ds * dt

            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
