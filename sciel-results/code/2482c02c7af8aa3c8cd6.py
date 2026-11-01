"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # p=16, split into 8 blocks of 2x2
    num_blocks = p // 2
    rank = num_blocks * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # u: coeffs for A (a, b, c, d) -> indices 0,1,2,3
    # v: coeffs for B (e, f, g, h) -> indices 0,1,2,3
    # w: coeffs for C (p11, p12, p21, p22) -> indices 0,1,2,3
    strassen_u = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, -1, 0] # M7
    ]
    strassen_v = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [-1, 0, 1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    strassen_w = [
        [1, 0, 0, 1], # M1 -> C00, C11
        [0, 0, 1, -1],# M2 -> C10, -C11
        [0, 1, 0, 1], # M3 -> C01, C11
        [1, 0, 1, 0], # M4 -> C00, C10
        [-1, 1, 0, 0],# M5 -> -C00, C01
        [0, 0, 0, 1], # M6 -> C11
        [1, 0, 0, 0]  # M7 -> C00
    ]
    
    for t in range(num_blocks):
        base_col = t * 7
        for s in range(7):
            col = base_col + s
            # Fill U
            # U index: i*m + j
            # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            U[0][col] = strassen_u[s][0] # i=0, j=0
            U[1][col] = strassen_u[s][1] # i=0, j=1
            U[2][col] = strassen_u[s][2] # i=1, j=0
            U[3][col] = strassen_u[s][3] # i=1, j=1
            
            # Fill V
            # V index: j*p + k
            # B block t has columns k_global = 2*t + k_local
            # B indices in block: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # j=0, k_local=0 -> k_global = 2*t
            # j=0, k_local=1 -> k_global = 2*t + 1
            # j=1, k_local=0 -> k_global = 2*t
            # j=1, k_local=1 -> k_global = 2*t + 1
            
            V[0 * p + (2*t + 0)][col] = strassen_v[s][0] # j=0, k=0
            V[0 * p + (2*t + 1)][col] = strassen_v[s][1] # j=0, k=1
            V[1 * p + (2*t + 0)][col] = strassen_v[s][2] # j=1, k=0
            V[1 * p + (2*t + 1)][col] = strassen_v[s][3] # j=1, k=1
            
            # Fill W
            # W index: k*n + i
            # C block t has columns k_global = 2*t + k_local
            # C indices in block: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # k_global=2*t, i=0 -> idx = (2*t)*2 + 0
            # k_global=2*t+1, i=0 -> idx = (2*t+1)*2 + 0
            # k_global=2*t, i=1 -> idx = (2*t)*2 + 1
            # k_global=2*t+1, i=1 -> idx = (2*t+1)*2 + 1
            
            W[(2*t + 0) * n + 0][col] = strassen_w[s][0] # k=0, i=0
            W[(2*t + 1) * n + 0][col] = strassen_w[s][1] # k=1, i=0
            W[(2*t + 0) * n + 1][col] = strassen_w[s][2] # k=0, i=1
            W[(2*t + 1) * n + 1][col] = strassen_w[s][3] # k=1, i=1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
