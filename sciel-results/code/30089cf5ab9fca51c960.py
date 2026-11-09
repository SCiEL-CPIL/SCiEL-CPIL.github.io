"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 189
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    products = [
        {'x': {(0,0): 1, (1,1): 1}, 'y': {(0,0): 1, (1,1): 1}, 'c': {(0,0): 1, (1,1): 1}},
        {'x': {(1,0): 1, (1,1): 1}, 'y': {(0,0): 1}, 'c': {(1,0): 1, (1,1): -1}},
        {'x': {(0,0): 1}, 'y': {(0,1): 1, (1,1): -1}, 'c': {(0,1): 1, (1,1): 1}},
        {'x': {(1,1): 1}, 'y': {(1,0): 1, (0,0): -1}, 'c': {(0,0): 1, (1,0): 1}},
        {'x': {(0,0): 1, (0,1): 1}, 'y': {(1,1): 1}, 'c': {(0,0): -1, (0,1): 1}},
        {'x': {(1,0): 1, (0,0): -1}, 'y': {(0,0): 1, (0,1): 1}, 'c': {(1,1): 1}},
        {'x': {(0,1): 1, (1,1): -1}, 'y': {(1,0): 1, (1,1): 1}, 'c': {(0,0): 1}}
    ]
    
    for prod in products:
        x_coeffs = prod['x']
        y_coeffs = prod['y']
        c_coeffs = prod['c']
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # U entries
                    for (a, b), xc in x_coeffs.items():
                        i_A = a * 3 + i
                        j_A = b * 3 + j
                        u_idx = i_A * m + j_A
                        U[u_idx][col] = xc
                    
                    # V entries
                    for (c, d), yc in y_coeffs.items():
                        i_B = c * 3 + j
                        j_B = d * 3 + k
                        v_idx = i_B * p + j_B
                        V[v_idx][col] = yc
                    
                    # W entries
                    for (e, f), cc in c_coeffs.items():
                        i_C = e * 3 + i
                        k_C = f * 3 + k
                        w_idx = k_C * n + i_C
                        W[w_idx][col] = cc
                        
                    col += 1
                    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
