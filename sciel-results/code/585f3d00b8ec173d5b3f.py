"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 7
    
    # Initialize matrices
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set indices based on i, j, k
    # u = i * m + j
    # v = j * p + k
    # w = k * n + i
    
    def set_u(i, j, r, val):
        U[i * m + j][r] = val
        
    def set_v(j, k, r, val):
        V[j * p + k][r] = val
        
    def set_w(k, i, r, val):
        W[k * n + i][r] = val

    # P1 = (a00 + a11)(b00 + b11)
    set_u(0, 0, 0, 1)
    set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1)
    set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1)
    set_w(1, 1, 0, 1)
    
    # P2 = (a10 + a11)b00
    set_u(1, 0, 1, 1)
    set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1)
    set_w(1, 1, 1, 1)
    
    # P3 = a00(b01 - b11)
    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1)
    set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1)
    set_w(1, 1, 2, -1)
    
    # P4 = a11(b10 - b00)
    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1)
    set_v(0, 0, 3, -1)
    set_w(0, 0, 3, -1)
    set_w(0, 1, 3, 1)
    
    # P5 = (a00 + a01)b11
    set_u(0, 0, 4, 1)
    set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(1, 0, 4, 1)
    set_w(1, 1, 4, -1)
    
    # P6 = (a10 - a00)(b00 + b01)
    set_u(1, 0, 5, 1)
    set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1)
    set_v(0, 1, 5, 1)
    set_w(0, 0, 5, 1)
    
    # P7 = (a01 - a11)(b10 + b11)
    set_u(0, 1, 6, 1)
    set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1)
    set_v(1, 1, 6, 1)
    set_w(0, 0, 6, -1)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
