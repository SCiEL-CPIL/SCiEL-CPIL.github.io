DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Pre-compute non-zero terms for each r
    # U_terms[r] = list of (coef, gi, gj)
    # V_terms[r] = list of (coef, gj, gk)
    # W_terms[r] = list of (coef, gi, gk)
    
    # Note: gi, gj depend on i_l, l, j_l.
    # So we can't fully pre-compute the global indices without the local ones.
    # But we can pre-compute the block offsets.
    
    # c[r] is dict (a,b) -> coef
    # d[r] is dict (b,c) -> coef
    # alpha[r] is dict (c,d) -> coef
    
    c_data = [None] * 8
    d_data = [None] * 8
    alpha_data = [None] * 8
    
    c_data[1] = {(1,1): 1, (2,2): 1}
    d_data[1] = {(1,1): 1, (2,2): 1}
    alpha_data[1] = {(1,1): 1, (2,2): 1}
    
    c_data[2] = {(2,1): 1, (2,2): 1}
    d_data[2] = {(1,1): 1}
    alpha_data[2] = {(1,1): -1, (2,1): 1}
    
    c_data[3] = {(1,1): 1}
    d_data[3] = {(1,2): 1, (2,2): -1}
    alpha_data[3] = {(1,2): 1, (2,2): -1}
    
    c_data[4] = {(2,2): 1}
    d_data[4] = {(2,1): 1, (1,1): -1}
    alpha_data[4] = {(1,1): 1, (2,1): 1}
    
    c_data[5] = {(1,1): 1, (1,2): 1}
    d_data[5] = {(2,2): 1}
    alpha_data[5] = {(1,2): 1, (2,2): 1}
    
    c_data[6] = {(2,1): 1, (1,1): -1}
    d_data[6] = {(1,1): 1, (1,2): 1}
    alpha_data[6] = {(1,1): 1}
    
    c_data[7] = {(1,2): 1, (2,2): -1}
    d_data[7] = {(2,1): 1, (2,2): 1}
    alpha_data[7] = {(2,2): 1}
    
    col = 0
    for r in range(1, 8):
        c_r = c_data[r]
        d_r = d_data[r]
        alpha_r = alpha_data[r]
        
        for i_l in range(3):
            for l in range(3):
                for j_l in range(3):
                    # U
                    for (a, b), coef in c_r.items():
                        gi = 3 * (a - 1) + i_l
                        gj = 3 * (b - 1) + l
                        U[gi * 6 + gj][col] += coef
                        
                    # V
                    for (b, cc), coef in d_r.items():
                        gj = 3 * (b - 1) + l
                        gk = 3 * (cc - 1) + j_l
                        V[gj * 6 + gk][col] += coef
                        
                    # W
                    for (cc, dd), coef in alpha_r.items():
                        gi = 3 * (cc - 1) + i_l
                        gk = 3 * (dd - 1) + j_l
                        W[gk * 6 + gi][col] += coef
                        
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
