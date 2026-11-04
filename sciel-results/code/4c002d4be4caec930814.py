import numpy as np

def construct():
    """
    Construct a rank-98 exact tensor decomposition for 4x4 * 4x8 matrix multiplication.
    """
    n, m, p = 4, 4, 8
    
    # Strassen coefficients for 2x2x2
    # alpha, beta, gamma, delta_u, delta_v, kappa as in the current program
    
    alpha = {
        1: {0: 1, 3: 1},
        2: {2: 1, 3: 1},
        3: {0: 1},
        4: {3: 1},
        5: {0: 1, 1: 1},
        6: {2: 1, 0: -1},
        7: {1: 1, 3: -1}
    }
    
    beta = {
        1: {0: 1, 3: 1},
        2: {0: 1},
        3: {1: 1, 3: -1},
        4: {2: 1, 0: -1},
        5: {3: 1},
        6: {0: 1, 1: 1},
        7: {2: 1, 3: 1}
    }
    
    gamma = {
        1: {(0,0): 1, (1,1): 1},
        2: {(1,0): 1, (1,1): -1},
        3: {(0,1): 1, (1,1): 1},
        4: {(0,0): 1, (1,0): 1},
        5: {(0,1): 1, (0,0): -1},
        6: {(1,1): 1},
        7: {(0,0): 1}
    }
    
    delta_u = {
        1: {(0,0): 1, (1,1): 1},
        2: {(1,0): 1, (1,1): 1},
        3: {(0,0): 1},
        4: {(1,1): 1},
        5: {(0,0): 1, (0,1): 1},
        6: {(1,0): 1, (0,0): -1},
        7: {(0,1): 1, (1,1): -1}
    }
    
    delta_v = {
        1: {(0,0): 1, (1,1): 1},
        2: {(0,0): 1},
        3: {(0,1): 1, (1,1): -1},
        4: {(1,0): 1, (0,0): -1},
        5: {(1,1): 1},
        6: {(0,0): 1, (0,1): 1},
        7: {(1,0): 1, (1,1): 1}
    }
    
    kappa = {
        1: {0: 1, 3: 1},
        2: {2: 1, 3: -1},
        3: {1: 1, 3: 1},
        4: {0: 1, 2: 1},
        5: {0: -1, 1: 1},
        6: {3: 1},
        7: {0: 1}
    }
    
    U444 = np.zeros((16, 49), dtype=float)
    V444 = np.zeros((16, 49), dtype=float)
    W444 = np.zeros((16, 49), dtype=float)
    
    for t in range(1, 8):
        for s in range(1, 8):
            r = (t - 1) * 7 + (s - 1)
            
            for a in range(2):
                for b in range(2):
                    coeff_u = delta_u[s].get((a,b), 0)
                    if coeff_u == 0: continue
                    for P, alpha_val in alpha[t].items():
                        coeff_total = coeff_u * alpha_val
                        if coeff_total == 0: continue
                        P_row = P // 2
                        P_col = P % 2
                        i = 2 * P_row + a
                        j = 2 * P_col + b
                        u_idx = i * 4 + j
                        U444[u_idx, r] += coeff_total
            
            for a in range(2):
                for b in range(2):
                    coeff_v = delta_v[s].get((a,b), 0)
                    if coeff_v == 0: continue
                    for Q, beta_val in beta[t].items():
                        coeff_total = coeff_v * beta_val
                        if coeff_total == 0: continue
                        Q_row = Q // 2
                        Q_col = Q % 2
                        j = 2 * Q_row + a
                        k = 2 * Q_col + b
                        v_idx = j * 4 + k
                        V444[v_idx, r] += coeff_total
            
            for a in range(2):
                for b in range(2):
                    coeff_m = gamma[s].get((a,b), 0)
                    if coeff_m == 0: continue
                    for PQ, kappa_val in kappa[t].items():
                        coeff_total = coeff_m * kappa_val
                        if coeff_total == 0: continue
                        PQ_row = PQ // 2
                        PQ_col = PQ % 2
                        i = 2 * PQ_row + a
                        k = 2 * PQ_col + b
                        w_idx = k * 4 + i
                        W444[w_idx, r] += coeff_total

    rank = 98
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((32, rank), dtype=float)
    W = np.zeros((32, rank), dtype=float)
    
    U[:, 0:49] = U444
    for j in range(4):
        for k in range(4):
            V[8*j + k, 0:49] = V444[4*j + k, :]
    for k in range(4):
        for i in range(4):
            W[4*k + i, 0:49] = W444[4*k + i, :]
            
    U[:, 49:98] = U444
    for j in range(4):
        for k in range(4, 8):
            V[8*j + k, 49:98] = V444[4*j + (k-4), :]
    for k in range(4, 8):
        for i in range(4):
            W[4*k + i, 49:98] = W444[4*(k-4) + i, :]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
