"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def set_u(r, i, j, val):
        U[i * m + j][r] = val
    def set_v(r, j, k, val):
        V[j * p + k][r] = val
    def set_w(r, k, i, val):
        W[k * n + i][r] = val

    # Rank 7 part for first two columns (k=0, 1)
    # r=1
    set_u(0, 0, 0, 1)
    set_v(0, 0, 0, 1)
    set_v(0, 1, 1, 1)
    set_w(0, 0, 0, 1)
    set_w(0, 1, 1, 1)
    
    # r=2
    set_u(1, 1, 0, 1)
    set_u(1, 1, 1, 1)
    set_v(1, 0, 0, 1)
    set_w(1, 0, 1, 1)
    set_w(1, 1, 1, 1)
    
    # r=3
    set_u(2, 0, 1, 1)
    set_v(2, 1, 0, 1)
    set_v(2, 1, 1, 1)
    set_w(2, 0, 1, 1)
    set_w(2, 1, 1, -1)
    
    # r=4
    set_u(3, 1, 1, 1)
    set_v(3, 0, 1, 1)
    set_v(3, 0, 0, -1)
    set_w(3, 0, 0, -1)
    set_w(3, 1, 0, 1)
    
    # r=5
    set_u(4, 0, 0, 1)
    set_u(4, 1, 0, 1)
    set_v(4, 0, 1, 1)
    set_v(4, 1, 1, 1)
    set_w(4, 0, 0, 1)
    
    # r=6
    set_u(5, 0, 1, 1)
    set_u(5, 1, 1, -1)
    set_v(5, 0, 0, 1)
    set_v(5, 1, 0, 1)
    set_w(5, 1, 1, 1)
    
    # r=7
    set_u(6, 0, 0, 1)
    set_u(6, 0, 1, -1)
    set_v(6, 1, 1, 1)
    set_v(6, 1, 0, -1)
    set_w(6, 0, 0, 1)
    set_w(6, 0, 1, 1)
    
    # Rank 4 part for third column (k=2)
    # r=8
    set_u(7, 0, 0, 1)
    set_v(7, 0, 2, 1)
    set_w(7, 2, 0, 1)
    
    # r=9
    set_u(8, 0, 1, 1)
    set_v(8, 1, 2, 1)
    set_w(8, 2, 0, 1)
    
    # r=10
    set_u(9, 1, 0, 1)
    set_v(9, 0, 2, 1)
    set_w(9, 2, 1, 1)
    
    # r=11
    set_u(10, 1, 1, 1)
    set_v(10, 1, 2, 1)
    set_w(10, 2, 1, 1)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
