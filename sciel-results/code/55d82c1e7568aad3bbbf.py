"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen coefficients for 2x2x2
    # u_s, v_s, w_s are lists of 4 elements corresponding to indices 0..3
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    u_coeffs = [
        [1, 0, 0, 1],  # s=1
        [0, 0, 1, 1],  # s=2
        [1, 0, 0, 0],  # s=3
        [0, 0, 0, 1],  # s=4
        [1, 1, 0, 0],  # s=5
        [-1, 0, 1, 0], # s=6
        [0, 1, 0, -1]  # s=7
    ]
    
    v_coeffs = [
        [1, 0, 0, 1],  # s=1
        [1, 0, 0, 0],  # s=2
        [0, 0, 1, -1], # s=3
        [-1, 1, 0, 0], # s=4
        [0, 0, 0, 1],  # s=5
        [1, 0, 1, 0],  # s=6
        [0, 1, 0, 1]   # s=7
    ]
    
    w_coeffs = [
        [1, 0, 0, 1],  # s=1
        [0, 0, 1, -1], # s=2
        [0, 1, 0, 1],  # s=3
        [1, 0, 1, 0],  # s=4
        [-1, 1, 0, 0], # s=5
        [0, 0, 0, 1],  # s=6
        [1, 0, 0, 0]   # s=7
    ]

    for t in range(3):
        for s_idx in range(7):
            r = 7 * t + s_idx
            u_s = u_coeffs[s_idx]
            v_s = v_coeffs[s_idx]
            w_s = w_coeffs[s_idx]
            
            # Fill U
            for i in range(n):
                for j in range(m):
                    U[i * m + j][r] = u_s[i * m + j]
                    
            # Fill V
            for j in range(m):
                for k in range(p):
                    if 2 * t <= k <= 2 * t + 1:
                        k_loc = k - 2 * t
                        V[j * p + k][r] = v_s[j * 2 + k_loc]
                    else:
                        V[j * p + k][r] = 0
                        
            # Fill W
            for k in range(p):
                for i in range(n):
                    if 2 * t <= k <= 2 * t + 1:
                        k_loc = k - 2 * t
                        W[k * n + i][r] = w_s[i * 2 + k_loc]
                    else:
                        W[k * n + i][r] = 0

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
