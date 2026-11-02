"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=10
    # We partition B into p/m = 10/2 = 5 blocks of size 2x2.
    num_blocks = p // m
    block_size = m  # 2
    
    # Strassen rank for 2x2 is 7
    strassen_rank = 7
    total_rank = num_blocks * strassen_rank
    
    # Initialize matrices
    U = [[0] * total_rank for _ in range(n * m)]  # 4 x 35
    V = [[0] * total_rank for _ in range(m * p)]  # 20 x 35
    W = [[0] * total_rank for _ in range(p * n)]  # 20 x 35
    
    # Strassen coefficients for A (rows 0..3 correspond to A[0,0], A[0,1], A[1,0], A[1,1])
    # Columns correspond to M1..M7
    # M1=(a+d)(e+h) -> a:1, d:1
    # M2=(c+d)e     -> c:1, d:1
    # M3=a(f-h)     -> a:1
    # M4=d(g-e)     -> d:1
    # M5=(a+b)h     -> a:1, b:1
    # M6=(c-a)(e+f) -> a:-1, c:1
    # M7=(b-d)(g+h) -> b:1, d:-1
    U_coeffs = [
        [1, 0, 1, 0, 1, -1, 0],  # a
        [0, 0, 0, 0, 1, 0, 1],   # b
        [0, 1, 0, 0, 0, 1, 0],   # c
        [1, 1, 0, 1, 0, 0, -1]   # d
    ]
    
    # Strassen coefficients for B (rows correspond to e,f,g,h)
    # M1=(a+d)(e+h) -> e:1, h:1
    # M2=(c+d)e     -> e:1
    # M3=a(f-h)     -> f:1, h:-1
    # M4=d(g-e)     -> g:1, h:-1
    # M5=(a+b)h     -> h:1
    # M6=(c-a)(e+f) -> e:1, f:1
    # M7=(b-d)(g+h) -> g:1, h:1
    V_coeffs = [
        [1, 1, 0, 0, 0, 1, 0],   # e
        [0, 0, 1, 0, 0, 1, 0],   # f
        [0, 0, 0, 1, 0, 0, 1],   # g
        [1, 0, -1, -1, 1, 0, 1]  # h
    ]
    
    # Strassen coefficients for C (rows correspond to C00, C10, C01, C11)
    # Note: W index is k*n + i. 
    # Block k produces C block at columns 2k, 2k+1.
    # C[0, 2k] is at W index (2k)*2 + 0 = 4k
    # C[1, 2k] is at W index (2k)*2 + 1 = 4k+1
    # C[0, 2k+1] is at W index (2k+1)*2 + 0 = 4k+2
    # C[1, 2k+1] is at W index (2k+1)*2 + 1 = 4k+3
    
    # C00 = M1 - M4 + M5 + M7
    # C10 = M2 + M4
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6
    W_coeffs = [
        [1, 0, 0, -1, 1, 0, 1],  # C00
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [1, -1, 1, 0, 0, 1, 0]   # C11
    ]
    
    for k in range(num_blocks):
        base_col = k * strassen_rank
        
        # Fill U
        for i in range(n * m):
            for r in range(strassen_rank):
                U[i][base_col + r] = U_coeffs[i][r]
        
        # Fill V
        # B is 2x10. Rows 0 and 1.
        # Block k uses columns 2k and 2k+1 of B.
        # V index for B[row, col] is row * p + col.
        # e = B[0, 2k] -> index 0 * 10 + 2k
        # f = B[0, 2k+1] -> index 0 * 10 + 2k + 1
        # g = B[1, 2k] -> index 1 * 10 + 2k
        # h = B[1, 2k+1] -> index 1 * 10 + 2k + 1
        
        v_indices = [
            0 * p + 2 * k,
            0 * p + 2 * k + 1,
            1 * p + 2 * k,
            1 * p + 2 * k + 1
        ]
        
        for idx_in_block, v_row in enumerate(v_indices):
            for r in range(strassen_rank):
                V[v_row][base_col + r] = V_coeffs[idx_in_block][r]
        
        # Fill W
        # W index for C[row, col] is col * n + row.
        # Block k produces C block at columns 2k and 2k+1.
        # C00 = C[0, 2k] -> index 2k * 2 + 0
        # C10 = C[1, 2k] -> index 2k * 2 + 1
        # C01 = C[0, 2k+1] -> index (2k+1) * 2 + 0
        # C11 = C[1, 2k+1] -> index (2k+1) * 2 + 1
        
        w_indices = [
            2 * k * n + 0,
            2 * k * n + 1,
            (2 * k + 1) * n + 0,
            (2 * k + 1) * n + 1
        ]
        
        for idx_in_block, w_row in enumerate(w_indices):
            for r in range(strassen_rank):
                W[w_row][base_col + r] = W_coeffs[idx_in_block][r]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
