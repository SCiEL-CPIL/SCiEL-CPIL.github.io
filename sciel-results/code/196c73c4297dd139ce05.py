"""Strassen-based decomposition for 2x3x10 matrix multiplication."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    rank = 55
    
    U = [[0] * rank for _ in range(n * m)]   # (6, 55)
    V = [[0] * rank for _ in range(m * p)]   # (30, 55)
    W = [[0] * rank for _ in range(p * n)]   # (20, 55)
    
    col = 0
    for block in range(5):
        k1 = block * 2
        k2 = block * 2 + 1
        
        # V indices for B rows 0,1,2 at columns k1,k2
        v_0_k1 = 0 * p + k1    # b11 equivalent
        v_0_k2 = 0 * p + k2    # b12 equivalent
        v_1_k1 = 1 * p + k1    # b21 equivalent
        v_1_k2 = 1 * p + k2    # b22 equivalent
        v_2_k1 = 2 * p + k1    # b31 equivalent
        v_2_k2 = 2 * p + k2    # b32 equivalent
        
        # W indices
        w_0_k1 = k1 * n + 0    # C[0][k1]
        w_0_k2 = k2 * n + 0    # C[0][k2]
        w_1_k1 = k1 * n + 1    # C[1][k1]
        w_1_k2 = k2 * n + 1    # C[1][k2]
        
        # U indices
        u_0_0 = 0 * m + 0  # a11
        u_0_1 = 0 * m + 1  # a12
        u_0_2 = 0 * m + 2  # a13
        u_1_0 = 1 * m + 0  # a21
        u_1_1 = 1 * m + 1  # a22
        u_1_2 = 1 * m + 2  # a23
        
        # M1 = (a11+a22)*(b1k1 + b2k1)  -- using b at row 0,1 col k1
        U[u_0_0][col] = 1
        U[u_1_1][col] = 1
        V[v_0_k1][col] = 1
        V[v_1_k1][col] = 1
        W[w_0_k1][col] = 1
        W[w_1_k2][col] = -1
        col += 1
        
        # M2 = (a21+a22)*b1k1
        U[u_1_0][col] = 1
        U[u_1_1][col] = 1
        V[v_0_k1][col] = 1
        W[w_1_k1][col] = 1
        W[w_0_k2][col] = -1
        col += 1
        
        # M3 = a11*(b1k1 - b2k1)
        U[u_0_0][col] = 1
        V[v_0_k1][col] = 1
        V[v_1_k1][col] = -1
        W[w_0_k1][col] = -1
        W[w_1_k2][col] = 1
        W[w_1_k1][col] = 1
        W[w_0_k2][col] = 1
        col += 1
        
        # M4 = a22*(b2k2 - b1k2)
        U[u_1_1][col] = 1
        V[v_1_k2][col] = 1
        V[v_0_k2][col] = -1
        W[w_0_k1][col] = -1
        W[w_1_k2][col] = 1
        W[w_1_k1][col] = 1
        W[w_0_k2][col] = 1
        col += 1
        
        # M5 = (a11+a12)*b2k2
        U[u_0_0][col] = 1
        U[u_0_1][col] = 1
        V[v_1_k2][col] = 1
        W[w_1_k1][col] = -1
        W[w_0_k2][col] = 1
        col += 1
        
        # M6 = (a21-a11)*(b1k1+b1k2)
        U[u_1_0][col] = 1
        U[u_0_0][col] = -1
        V[v_0_k1][col] = 1
        V[v_0_k2][col] = 1
        W[w_0_k1][col] = 1
        W[w_1_k2][col] = 1
        W[w_1_k1][col] = 1
        W[w_0_k2][col] = 1
        col += 1
        
        # M7 = (a12-a22)*(b2k1+b2k2)
        U[u_0_1][col] = 1
        U[u_1_1][col] = -1
        V[v_1_k1][col] = 1
        V[v_1_k2][col] = 1
        W[w_0_k1][col] = 1
        W[w_1_k2][col] = -1
        W[w_1_k1][col] = 1
        W[w_0_k2][col] = -1
        col += 1
        
        # Cross terms
        # a13*b3k1 -> C[0][k1]
        U[u_0_2][col] = 1
        V[v_2_k1][col] = 1
        W[w_0_k1][col] = 1
        col += 1
        
        # a13*b3k2 -> C[0][k2]
        U[u_0_2][col] = 1
        V[v_2_k2][col] = 1
        W[w_0_k2][col] = 1
        col += 1
        
        # a23*b3k1 -> C[1][k1]
        U[u_1_2][col] = 1
        V[v_2_k1][col] = 1
        W[w_1_k1][col] = 1
        col += 1
        
        # a23*b3k2 -> C[1][k2]
        U[u_1_2][col] = 1
        V[v_2_k2][col] = 1
        W[w_1_k2][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
