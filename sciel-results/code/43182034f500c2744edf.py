import numpy as np

def construct():
    n, m, p = 2, 4, 8
    rank = 56
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Local Strassen matrices (4x7)
    # U_loc[j_loc, c]
    U_loc = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    
    # V_loc[j_loc*2 + k_loc, c]
    V_loc = [
        [1, 1, 0, -1, 0, 1, 0], # j=0, k=0
        [0, 0, 0, 1, 0, 1, 0],  # j=0, k=1
        [0, 0, 1, 0, 0, 0, 1],  # j=1, k=0
        [1, 0, -1, 0, 1, 0, 1]  # j=1, k=1
    ]
    
    # W_loc[k_loc*2 + i, c]
    W_loc = [
        [1, 0, 0, 1, -1, 0, 1], # k=0, i=0
        [0, 0, 1, 0, 1, 0, 0],  # k=0, i=1
        [0, 1, 0, 1, 0, 0, 0],  # k=1, i=0
        [1, -1, 1, 0, 0, 1, 0]  # k=1, i=1
    ]
    
    col_offset = 0
    for g_j in range(2):
        for g_k in range(4):
            for c in range(7):
                c_global = col_offset + c
                
                for i in range(2):
                    for j_loc in range(2):
                        j_global = 2 * g_j + j_loc
                        r_U = i * m + j_global
                        U[r_U, c_global] = U_loc[j_loc, c]
                        
                        r_V = j_global * p + (2 * g_k + 0) # Placeholder, V needs k
                        # V depends on k_loc too, handled in next loop or combined
                        
                # Better to iterate over all local indices for V and W
                
                # Fill V
                for j_loc in range(2):
                    j_global = 2 * g_j + j_loc
                    for k_loc in range(2):
                        k_global = 2 * g_k + k_loc
                        r_V = j_global * p + k_global
                        V[r_V, c_global] = V_loc[j_loc * 2 + k_loc, c]
                        
                # Fill W
                for k_loc in range(2):
                    k_global = 2 * g_k + k_loc
                    for i in range(2):
                        r_W = k_global * n + i
                        W[r_W, c_global] = W_loc[k_loc * 2 + i, c]
                        
            col_offset += 7
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
