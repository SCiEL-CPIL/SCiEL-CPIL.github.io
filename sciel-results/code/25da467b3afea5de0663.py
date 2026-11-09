import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    rank = 97
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen part for 4x4x4
    # Macro coefficients c_s[I, J] for X_s = sum c_s[I,J] A_{I,J}
    # d_s[J, K] for Y_s = sum d_s[J,K] B_{J,K}
    # e_s[I, J] for C contribution: C_{I,J} += e_s[I,J] P_s
    
    macro_c = {
        1: {(0,0): 1, (1,1): 1},
        2: {(1,0): 1, (1,1): 1},
        3: {(0,0): 1},
        4: {(1,1): 1},
        5: {(0,0): 1, (0,1): 1},
        6: {(1,0): 1, (0,0): -1},
        7: {(0,1): 1, (1,1): -1}
    }
    
    macro_d = {
        1: {(0,0): 1, (1,1): 1},
        2: {(0,0): 1},
        3: {(0,1): 1, (1,1): -1},
        4: {(1,0): 1, (0,0): -1},
        5: {(1,1): 1},
        6: {(0,0): 1, (0,1): 1},
        7: {(1,0): 1, (1,1): 1}
    }
    
    micro_u = {
        1: {(0,0): 1, (1,1): 1},
        2: {(1,0): 1, (1,1): 1},
        3: {(0,0): 1},
        4: {(1,1): 1},
        5: {(0,0): 1, (0,1): 1},
        6: {(1,0): 1, (0,0): -1},
        7: {(0,1): 1, (1,1): -1}
    }
    
    micro_v = {
        1: {(0,0): 1, (1,1): 1},
        2: {(0,0): 1},
        3: {(0,1): 1, (1,1): -1},
        4: {(1,0): 1, (0,0): -1},
        5: {(1,1): 1},
        6: {(0,0): 1, (0,1): 1},
        7: {(1,0): 1, (1,1): 1}
    }
    
    micro_w = {
        1: {(0,0): 1, (1,1): 1},
        2: {(1,0): 1, (1,1): -1},
        3: {(0,1): 1, (1,1): 1},
        4: {(0,0): 1, (1,0): 1},
        5: {(0,0): -1, (0,1): 1},
        6: {(1,1): 1},
        7: {(0,0): 1}
    }
    
    macro_e = {
        1: {(0,0): 1, (1,1): 1},
        2: {(1,0): 1, (1,1): -1},
        3: {(0,1): 1, (1,1): 1},
        4: {(0,0): 1, (1,0): 1},
        5: {(0,0): -1, (0,1): 1},
        6: {(1,1): 1},
        7: {(0,0): 1}
    }
    
    for s in range(1, 8):
        c_s = macro_c[s]
        d_s = macro_d[s]
        e_s = macro_e[s]
        
        for k in range(1, 8):
            u_k = micro_u[k]
            v_k = micro_v[k]
            w_k = micro_w[k]
            
            for (ip, jp), coeff_u in u_k.items():
                for (I, J), coeff_c in c_s.items():
                    i = 2 * I + ip
                    j = 2 * J + jp
                    U[i * 4 + j][col] += coeff_u * coeff_c
            
            for (ip, jp), coeff_v in v_k.items():
                for (J, K), coeff_d in d_s.items():
                    j = 2 * J + ip
                    k = 2 * K + jp
                    V[j * 7 + k][col] += coeff_v * coeff_d
            
            for (ip, jp), coeff_w in w_k.items():
                for (I, J), coeff_e in e_s.items():
                    i = 2 * I + ip
                    k = 2 * J + jp
                    W[k * 4 + i][col] += coeff_w * coeff_e
            
            col += 1
    
    # Schoolbook part for 4x4x3 (cols 4-6 of B)
    for k in range(4, 7):
        for i in range(4):
            for j in range(4):
                U[i * 4 + j][col] = 1
                V[j * 7 + k][col] = 1
                W[k * 4 + i][col] = 1
                col += 1
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
