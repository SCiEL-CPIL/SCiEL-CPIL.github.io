"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 11 decomposition: Strassen (7) + Schoolbook (4)
    rank = 11
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set a column
    def set_col(idx, u_vals, v_vals, w_vals):
        for i in range(n * m):
            U[i][idx] = u_vals[i]
        for i in range(m * p):
            V[i][idx] = v_vals[i]
        for i in range(p * n):
            W[i][idx] = w_vals[i]
            
    # Strassen terms for columns 0,1 of B
    # U rows: A00, A01, A10, A11
    # V rows: B00, B01, B02, B10, B11, B12
    # W rows: C00, C10, C01, C11, C02, C12
    
    # P1
    set_col(0, 
            [1, 0, 0, 1], 
            [1, 0, 0, 0, 1, 0], 
            [1, 0, 1, -1, 0, 0])
            
    # P2
    set_col(1, 
            [0, 0, 1, 1], 
            [1, 0, 0, 0, 0, 0], 
            [0, 1, 0, 1, 0, 0])
            
    # P3
    set_col(2, 
            [1, 0, 0, 0], 
            [0, 1, 0, 0, -1, 0], 
            [0, 0, -1, 1, 0, 0])
            
    # P4
    set_col(3, 
            [0, 0, 0, 1], 
            [-1, 0, 0, 1, 0, 0], 
            [1, 1, -1, 0, 0, 0])
            
    # P5
    set_col(4, 
            [1, 1, 0, 0], 
            [0, 0, 0, 0, 1, 0], 
            [-1, 0, 0, 0, 0, 0])
            
    # P6
    set_col(5, 
            [0, -1, 1, 0], 
            [1, 1, 0, 0, 0, 0], 
            [0, 0, 0, 1, 0, 0])
            
    # P7
    set_col(6, 
            [1, 0, 0, -1], 
            [0, 0, 0, 1, 1, 0], 
            [1, -1, 1, 0, 0, 0])
            
    # Schoolbook terms for column 2 of B
    # P8: A00 * B02 -> C02
    set_col(7, 
            [1, 0, 0, 0], 
            [0, 0, 1, 0, 0, 0], 
            [0, 0, 0, 0, 1, 0])
            
    # P9: A01 * B12 -> C02
    set_col(8, 
            [0, 1, 0, 0], 
            [0, 0, 0, 0, 0, 1], 
            [0, 0, 0, 0, 1, 0])
            
    # P10: A10 * B02 -> C12
    set_col(9, 
            [0, 0, 1, 0], 
            [0, 0, 1, 0, 0, 0], 
            [0, 0, 0, 0, 0, 1])
            
    # P11: A11 * B12 -> C12
    set_col(10, 
            [0, 0, 0, 1], 
            [0, 0, 0, 0, 0, 1], 
            [0, 0, 0, 0, 0, 1])

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
