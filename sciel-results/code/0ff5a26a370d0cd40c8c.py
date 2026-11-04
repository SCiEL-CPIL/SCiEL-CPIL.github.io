"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 using 8 Strassen blocks
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Precompute Strassen factors for 2x2x2
    # U_t, V_t, W_t are 2x2 matrices
    # Indices: U[i,j], V[j,k], W[k,i]
    
    # M1
    U1 = [[1, 0], [0, 1]]
    V1 = [[1, 0], [0, 1]]
    W1 = [[1, 0], [0, 1]] # W[k,i]: W[0,0]=1, W[1,1]=1
    
    # M2
    U2 = [[0, 0], [1, 1]]
    V2 = [[1, 0], [0, 0]]
    W2 = [[0, 1], [0, -1]] # W[0,1]=1, W[1,1]=-1
    
    # M3
    U3 = [[1, 0], [0, 0]]
    V3 = [[0, 1], [0, -1]] # B[0,1] - B[1,1]
    W3 = [[0, 0], [1, 1]] # W[1,0]=1, W[1,1]=1
    
    # M4
    U4 = [[0, 0], [0, 1]]
    V4 = [[-1, 0], [1, 0]] # B[1,0] - B[0,0]
    W4 = [[1, 1], [0, 0]] # W[0,0]=1, W[0,1]=1
    
    # M5
    U5 = [[1, 1], [0, 0]]
    V5 = [[0, 0], [0, 1]]
    W5 = [[-1, 0], [1, 0]] # W[0,0]=-1, W[1,0]=1
    
    # M6
    U6 = [[-1, 0], [1, 0]]
    V6 = [[1, 1], [0, 0]]
    W6 = [[0, 0], [0, 1]] # W[1,1]=1
    
    # M7
    U7 = [[0, 1], [-1, 0]]
    V7 = [[0, 0], [1, 1]]
    W7 = [[1, 0], [0, 0]] # W[0,0]=1
    
    factors = [
        (U1, V1, W1),
        (U2, V2, W2),
        (U3, V3, W3),
        (U4, V4, W4),
        (U5, V5, W5),
        (U6, V6, W6),
        (U7, V7, W7)
    ]
    
    col = 0
    # 8 blocks: j_block in {0,1}, k_block in {0,1,2,3}
    for j_block in range(2):
        for k_block in range(4):
            for U_t, V_t, W_t in factors:
                # Map local indices to global
                # i_local in {0,1} -> i_global
                # j_local in {0,1} -> j_global = 2*j_block + j_local
                # k_local in {0,1} -> k_global = 2*k_block + k_local
                
                for i_l in range(2):
                    for j_l in range(2):
                        u_val = U_t[i_l][j_l]
                        if u_val != 0:
                            i_g = i_l
                            j_g = 2 * j_block + j_l
                            U[i_g * m + j_g][col] = u_val
                            
                for j_l in range(2):
                    for k_l in range(2):
                        v_val = V_t[j_l][k_l]
                        if v_val != 0:
                            j_g = 2 * j_block + j_l
                            k_g = 2 * k_block + k_l
                            V[j_g * p + k_g][col] = v_val
                            
                for k_l in range(2):
                    for i_l in range(2):
                        w_val = W_t[k_l][i_l]
                        if w_val != 0:
                            k_g = 2 * k_block + k_l
                            i_g = i_l
                            W[k_g * n + i_g][col] = w_val
                            
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
