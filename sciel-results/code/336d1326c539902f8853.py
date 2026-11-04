"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition: 6x6 = 3x3 blocks of 2x2.
    # Rank = (3*3*3) * 7 = 189.
    rank = 189
    
    # Initialize U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 2x2 Strassen decomposition vectors.
    # Indices for 2x2:
    # u_idx = 2*i + j
    # v_idx = 2*j + k
    # w_idx = 2*k + i
    # l is the multiplication index (0..6)
    
    # U_inn[l] is a dict: {(i, j): val}
    # V_inn[l] is a dict: {(j, k): val}
    # W_inn[l] is a dict: {(k, i): val}
    
    U_inn = [
        {(0, 0): 1, (1, 1): 1},  # M1
        {(1, 0): 1, (1, 1): 1},  # M2
        {(0, 0): 1},             # M3
        {(1, 1): 1},             # M4
        {(0, 0): 1, (1, 0): 1},  # M5
        {(0, 1): 1, (0, 0): -1}, # M6
        {(1, 1): 1, (0, 1): -1}  # M7
    ]
    
    V_inn = [
        {(0, 0): 1, (1, 1): 1},  # M1
        {(0, 0): 1},             # M2
        {(0, 1): 1, (1, 1): -1}, # M3
        {(1, 0): 1, (0, 0): -1}, # M4
        {(1, 1): 1},             # M5
        {(0, 0): 1, (1, 0): 1},  # M6
        {(0, 1): 1, (1, 1): 1}   # M7
    ]
    
    W_inn = [
        {(0, 0): 1, (1, 1): 1},  # M1
        {(1, 0): 1},             # M2
        {(0, 1): 1, (1, 1): 1},  # M3
        {(0, 0): 1, (1, 0): 1},  # M4
        {(0, 0): -1, (0, 1): 1}, # M5
        {(1, 1): 1},             # M6
        {(0, 0): 1, (1, 1): 1}   # M7
    ]
    
    # Iterate over block indices I, J, K (0..2) and Strassen index l (0..6)
    for I in range(3):
        for J in range(3):
            for K in range(3):
                for l in range(7):
                    col = (I * 3 + J) * 3 * 7 + K * 7 + l
                    
                    # Fill U
                    # u_row = i_glob * m + j_glob
                    # i_glob = 2*I + i
                    # j_glob = 2*J + j
                    for (i, j), val in U_inn[l].items():
                        i_glob = 2 * I + i
                        j_glob = 2 * J + j
                        u_row = i_glob * m + j_glob
                        U[u_row][col] += val
                        
                    # Fill V
                    # v_row = j_glob * p + k_glob
                    # j_glob = 2*J + j
                    # k_glob = 2*K + k
                    for (j, k), val in V_inn[l].items():
                        j_glob = 2 * J + j
                        k_glob = 2 * K + k
                        v_row = j_glob * p + k_glob
                        V[v_row][col] += val
                        
                    # Fill W
                    # w_row = k_glob * n + i_glob
                    # k_glob = 2*K + k
                    # i_glob = 2*I + i
                    for (k, i), val in W_inn[l].items():
                        k_glob = 2 * K + k
                        i_glob = 2 * I + i
                        w_row = k_glob * n + i_glob
                        W[w_row][col] += val
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
