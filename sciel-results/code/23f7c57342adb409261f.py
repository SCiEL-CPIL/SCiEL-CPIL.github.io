import time

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 16
    num_blocks = p // 2  # 8
    rank = 7 * num_blocks  # 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for block in range(num_blocks):
        base_col = block * 7
        b0 = 2 * block
        b1 = 2 * block + 1
        
        v_b00 = 0 * p + b0
        v_b01 = 0 * p + b1
        v_b10 = 1 * p + b0
        v_b11 = 1 * p + b1
        
        w_c00 = b0 * n + 0
        w_c01 = b1 * n + 0
        w_c10 = b0 * n + 1
        w_c11 = b1 * n + 1
        
        # M1 = (a+b)(e+f)
        col = base_col
        U[0][col] = 1; U[1][col] = 1
        V[v_b00][col] = 1; V[v_b01][col] = 1
        W[w_c00][col] = 1; W[w_c11][col] = 1
        
        # M2 = (c+d)e
        col = base_col + 1
        U[2][col] = 1; U[3][col] = 1
        V[v_b00][col] = 1
        W[w_c10][col] = 1; W[w_c11][col] = -1
        
        # M3 = a(f-h)
        col = base_col + 2
        U[0][col] = 1
        V[v_b01][col] = 1; V[v_b11][col] = -1
        W[w_c01][col] = 1; W[w_c11][col] = -1
        
        # M4 = d(g-e)
        col = base_col + 3
        U[3][col] = 1
        V[v_b10][col] = 1; V[v_b00][col] = -1
        W[w_c00][col] = -1; W[w_c11][col] = 1
        
        # M5 = (a+d)(h-g)
        col = base_col + 4
        U[0][col] = 1; U[3][col] = 1
        V[v_b11][col] = 1; V[v_b10][col] = -1
        W[w_c00][col] = 1; W[w_c10][col] = 1
        
        # M6 = (b+c)g (not used in output)
        col = base_col + 5
        U[1][col] = 1; U[2][col] = 1
        V[v_b10][col] = 1
        
        # M7 = (b+d)(f-g)
        col = base_col + 6
        U[1][col] = 1; U[3][col] = 1
        V[v_b01][col] = 1; V[v_b10][col] = -1
        W[w_c00][col] = -1; W[w_c01][col] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
