"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    # Strassen decomposition for 4x4x4 (rank 49) + 4x4x1 (rank 1) = rank 50
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 2x2 Strassen coefficients
    # u[i, j, k] is coefficient of A[i, j] in S_k
    # v[i, j, k] is coefficient of B[i, j] in T_k
    # w[i, j, k] is coefficient of P_k in C[i, j]
    u_2x2 = [
        [[1, 0, 0, 0, 1, -1, 0], [0, 0, 0, 0, 1, 0, 1]], 
        [[0, 1, 0, 0, 0, 1, -1], [1, 1, 0, 0, 0, 0, 0]]  
    ]
    v_2x2 = [
        [[1, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1]], 
        [[1, 0, -1, 1, 0, 0, 1], [0, 0, 0, 0, 1, 1, 1]]  
    ]
    w_2x2 = [
        [[1, 0, 0, 1, -1, 0, 1], [0, 0, 1, 0, 1, 0, 0]], 
        [[0, 1, 0, 1, 0, 0, 0], [1, -1, 1, 0, 0, 1, 0]]  
    ]
    
    # 4x4 Strassen coefficients
    # s[a, b, k] is coefficient of A[a, b] in S_k
    # t[a, b, k] is coefficient of B[a, b] in T_k
    s_4x4 = [
        [[1, 0, 0, 0, 1, -1, 0], [0, 0, 0, 0, 1, 0, 1]], 
        [[0, 1, 0, 0, 0, 1, -1], [1, 1, 0, 0, 0, 0, 0]]  
    ]
    t_4x4 = [
        [[1, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1]], 
        [[1, 0, -1, 1, 0, 0, 1], [0, 0, 0, 0, 1, 1, 1]]  
    ]
    c_4x4 = [
        [[1, 0, 0, 1, -1, 0, 1], [0, 0, 1, 0, 1, 0, 0]], 
        [[0, 1, 0, 1, 0, 0, 0], [1, -1, 1, 0, 0, 1, 0]]  
    ]
    
    col = 0
    for k_prime in range(7):
        for l in range(7):
            for a in range(2):
                for b in range(2):
                    for i_prime in range(2):
                        for j_prime in range(2):
                            u_idx = (2 * a + i_prime) * 4 + (2 * b + j_prime)
                            U[u_idx][col] += s_4x4[a][b][k_prime] * u_2x2[i_prime][j_prime][l]
                            
                            v_idx = (2 * a + i_prime) * 5 + (2 * b + j_prime)
                            V[v_idx][col] += t_4x4[a][b][k_prime] * v_2x2[i_prime][j_prime][l]
                            
                            w_idx = (2 * b + j_prime) * 4 + (2 * a + i_prime)
                            W[w_idx][col] += c_4x4[a][b][k_prime] * w_2x2[i_prime][j_prime][l]
            col += 1
            
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][49] = 1
            V[j * 5 + 4][49] = 1
            W[4 * 4 + i][49] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
