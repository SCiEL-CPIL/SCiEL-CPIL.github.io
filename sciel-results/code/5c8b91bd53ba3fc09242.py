"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 10
    rank = 36
    U_mat = [[0] * rank for _ in range(n * m)] # 6 x 36
    V_mat = [[0] * rank for _ in range(m * p)] # 30 x 36
    W_mat = [[0] * rank for _ in range(p * n)] # 20 x 36
    
    # Strassen for j=0,1 (5 blocks of 2x2)
    for l in range(5):
        k0 = 2 * l
        k1 = 2 * l + 1
        
        # A indices
        a00 = 0 # 0
        a01 = 1 # 1
        a10 = 3 # 3
        a11 = 4 # 4
        
        # B indices
        b0k0 = 0 * 10 + k0
        b0k1 = 0 * 10 + k1
        b1k0 = 1 * 10 + k0
        b1k1 = 1 * 10 + k1
        
        # W indices
        c0k0 = k0 * 2 + 0
        c0k1 = k1 * 2 + 0
        c1k0 = k0 * 2 + 1
        c1k1 = k1 * 2 + 1
        
        # P1 = (a00 + a11) * (b0k0 + b1k1)
        idx = 7 * l
        U_mat[a00][idx] += 1
        U_mat[a11][idx] += 1
        V_mat[b0k0][idx] += 1
        V_mat[b1k1][idx] += 1
        W_mat[c0k0][idx] += 1
        W_mat[c1k1][idx] += 1
        
        # P2 = (a10 + a11) * b0k0
        idx = 7 * l + 1
        U_mat[a10][idx] += 1
        U_mat[a11][idx] += 1
        V_mat[b0k0][idx] += 1
        W_mat[c1k0][idx] += 1
        W_mat[c1k1][idx] -= 1
        
        # P3 = a00 * (b0k1 - b1k1)
        idx = 7 * l + 2
        U_mat[a00][idx] += 1
        V_mat[b0k1][idx] += 1
        V_mat[b1k1][idx] -= 1
        W_mat[c0k1][idx] += 1
        W_mat[c1k1][idx] += 1
        
        # P4 = a11 * (b1k0 - b0k0)
        idx = 7 * l + 3
        U_mat[a11][idx] += 1
        V_mat[b1k0][idx] += 1
        V_mat[b0k0][idx] -= 1
        W_mat[c0k0][idx] -= 1
        W_mat[c1k0][idx] += 1
        
        # P5 = (a00 + a01) * b1k1
        idx = 7 * l + 4
        U_mat[a00][idx] += 1
        U_mat[a01][idx] += 1
        V_mat[b1k1][idx] += 1
        W_mat[c0k0][idx] -= 1
        W_mat[c0k1][idx] += 1
        
        # P6 = (a10 - a00) * (b0k0 + b0k1)
        idx = 7 * l + 5
        U_mat[a10][idx] += 1
        U_mat[a00][idx] -= 1
        V_mat[b0k0][idx] += 1
        V_mat[b0k1][idx] += 1
        W_mat[c1k1][idx] -= 1
        
        # P7 = (a01 - a11) * (b1k0 + b1k1)
        idx = 7 * l + 6
        U_mat[a01][idx] += 1
        U_mat[a11][idx] -= 1
        V_mat[b1k0][idx] += 1
        V_mat[b1k1][idx] += 1
        W_mat[c0k0][idx] += 1
        W_mat[c1k1][idx] += 1
        
    # Schoolbook for j=2 (Rank 1)
    idx = 35
    for i in range(2):
        U_mat[i * 3 + 2][idx] += 1
    for k in range(10):
        V_mat[2 * 10 + k][idx] += 1
    for i in range(2):
        for k in range(10):
            W_mat[k * 2 + i][idx] += 1
            
    return {'U': U_mat, 'V': V_mat, 'W': W_mat}
# EVOLVE-BLOCK-END
