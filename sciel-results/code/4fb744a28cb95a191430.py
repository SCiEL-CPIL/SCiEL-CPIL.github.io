"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 8 * 7 = 56
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen components for 2x2x2
    # U_local: 4 entries (i*m+j)
    # V_local: 4 entries (j*p_local + k_local) -> here p_local=2
    # W_local: 4 entries (k_local*n + i)
    
    comps = [
        # U, V, W
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # M3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # M5
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, -1]), # M7
    ]
    
    for l in range(8):
        for c in range(7):
            col = l * 7 + c
            u_loc, v_loc, w_loc = comps[c]
            
            # Fill U
            for idx in range(4):
                U[idx][col] = u_loc[idx]
                
            # Fill V
            # Local V index j*2 + k'
            for j in range(2):
                for kp in range(2):
                    local_idx = j * 2 + kp
                    global_v_idx = j * p + (2 * l + kp)
                    V[global_v_idx][col] = v_loc[local_idx]
                    
            # Fill W
            # Local W index k'*n + i
            for kp in range(2):
                for i in range(2):
                    local_idx = kp * n + i
                    global_w_idx = (2 * l + kp) * n + i
                    W[global_w_idx][col] = w_loc[local_idx]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
