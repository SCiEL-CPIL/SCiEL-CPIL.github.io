"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Partition p columns into blocks of 2
    block_size = 2
    num_blocks = p // block_size
    
    # Strassen rank for 2x2x2 is 7
    rank_per_block = 7
    rank = num_blocks * rank_per_block
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen terms for 2x2x2
    # Each term is (u_vec, v_vec, w_vec)
    # u_vec: coeffs for A[0,0], A[0,1], A[1,0], A[1,1]
    # v_vec: coeffs for B[0,0], B[0,1], B[1,0], B[1,1]
    # w_vec: coeffs for C[0,0], C[1,0], C[0,1], C[1,1] (index k*n+i)
    
    terms = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        # M3
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 0, 1, 1]),
        # M4
        ([0, 0, 0, 1], [-1, 1, 0, 0], [1, 1, 0, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        # M6
        ([-1, 0, 1, 0], [1, 0, 1, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, 0, -1], [0, 1, 0, 1], [1, 0, 0, 0]),
    ]
    
    for b in range(num_blocks):
        for t in range(rank_per_block):
            col = b * rank_per_block + t
            u, v, w = terms[t]
            
            # Fill U
            # U index: i*m + j
            # u is [A00, A01, A10, A11]
            U[0][col] = u[0] # A00
            U[1][col] = u[1] # A01
            U[2][col] = u[2] # A10
            U[3][col] = u[3] # A11
            
            # Fill V
            # V index: j*p + k_global
            # v is [B00, B01, B10, B11]
            # B00: j=0, k_local=0 -> k_global = b*2 + 0
            V[0 * p + (b * block_size + 0)][col] = v[0]
            # B01: j=0, k_local=1 -> k_global = b*2 + 1
            V[0 * p + (b * block_size + 1)][col] = v[1]
            # B10: j=1, k_local=0 -> k_global = b*2 + 0
            V[1 * p + (b * block_size + 0)][col] = v[2]
            # B11: j=1, k_local=1 -> k_global = b*2 + 1
            V[1 * p + (b * block_size + 1)][col] = v[3]
            
            # Fill W
            # W index: k_global*n + i
            # w is [C00, C10, C01, C11]
            # C00: i=0, k_local=0 -> k_global = b*2 + 0
            W[(b * block_size + 0) * n + 0][col] = w[0]
            # C10: i=1, k_local=0 -> k_global = b*2 + 0
            W[(b * block_size + 0) * n + 1][col] = w[1]
            # C01: i=0, k_local=1 -> k_global = b*2 + 1
            W[(b * block_size + 1) * n + 0][col] = w[2]
            # C11: i=1, k_local=1 -> k_global = b*2 + 1
            W[(b * block_size + 1) * n + 1][col] = w[3]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
