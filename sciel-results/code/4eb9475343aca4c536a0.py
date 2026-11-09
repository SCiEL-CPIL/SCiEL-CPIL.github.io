"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Define the 7 vectors for Strassen 2x2x2 decomposition
    # Indices for 2x2 matrix: 00->0, 01->1, 10->2, 11->3
    U_2 = np.array([
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ])
    
    V_2 = np.array([
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1]   # M7
    ])
    
    W_2 = np.array([
        [1, 0, 0, 1],  # M1
        [0, 0, 1, -1], # M2
        [0, 1, 0, 1],  # M3
        [1, 0, 1, 0],  # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1],  # M6
        [1, 0, 0, 0]   # M7
    ])
    
    rank = 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col_idx = 0
    
    # 4 blocks of output columns (each of width 2)
    for b in range(4):
        # Part 1: A_L * B_T_b
        # A_L: i in {0,1}, j in {0,1}
        # B_T_b: j in {0,1}, k in {2b, 2b+1}
        
        for s in range(7):
            u_vec = U_2[s]
            v_vec = V_2[s]
            w_vec = W_2[s]
            
            # Fill U for A_L
            for i in range(2):
                for j in range(2):
                    # Local index i*2+j
                    val = u_vec[i * 2 + j]
                    if val != 0:
                        # Global U index: i*m + j = i*4 + j
                        U[i * 4 + j, col_idx] = val
                        
            # Fill V for B_T_b
            for j in range(2):
                for k_loc in range(2):
                    # Local index j*2+k_loc
                    val = v_vec[j * 2 + k_loc]
                    if val != 0:
                        # Global V index: j*p + k = j*8 + (2b + k_loc)
                        V[j * 8 + (2 * b + k_loc), col_idx] = val
                        
            # Fill W for C_i,k
            for i in range(2):
                for k_loc in range(2):
                    # Local index k_loc*2 + i
                    val = w_vec[k_loc * 2 + i]
                    if val != 0:
                        # Global W index: k*n + i = (2b + k_loc)*2 + i
                        W[(2 * b + k_loc) * 2 + i, col_idx] = val
                        
            col_idx += 1
            
        # Part 2: A_R * B_B_b
        # A_R: i in {0,1}, j in {2,3}
        # B_B_b: j in {2,3}, k in {2b, 2b+1}
        
        for s in range(7):
            u_vec = U_2[s]
            v_vec = V_2[s]
            w_vec = W_2[s]
            
            # Fill U for A_R
            for i in range(2):
                for j_loc in range(2):
                    # Local index i*2+j_loc
                    val = u_vec[i * 2 + j_loc]
                    if val != 0:
                        # Global j = 2 + j_loc
                        # Global U index: i*4 + (2 + j_loc)
                        U[i * 4 + (2 + j_loc), col_idx] = val
                        
            # Fill V for B_B_b
            for j_loc in range(2):
                for k_loc in range(2):
                    # Local index j_loc*2+k_loc
                    val = v_vec[j_loc * 2 + k_loc]
                    if val != 0:
                        # Global j = 2 + j_loc
                        # Global k = 2b + k_loc
                        # Global V index: (2 + j_loc)*8 + (2b + k_loc)
                        V[(2 + j_loc) * 8 + (2 * b + k_loc), col_idx] = val
                        
            # Fill W for C_i,k
            for i in range(2):
                for k_loc in range(2):
                    # Local index k_loc*2 + i
                    val = w_vec[k_loc * 2 + i]
                    if val != 0:
                        # Global W index: (2b + k_loc)*2 + i
                        W[(2 * b + k_loc) * 2 + i, col_idx] = val
                        
            col_idx += 1

    # Convert to lists of lists as required by interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
