"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Split m into two parts of size 2
    m1 = 2
    m2 = 2
    
    rank = 20 # 10 + 10
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a schoolbook decomposition for a sub-block
    # A_sub is n x m_sub, B_sub is m_sub x p
    # U_sub indices: i in [0,n), j in [0, m_sub)
    # V_sub indices: j in [0, m_sub), k in [0, p)
    # W_sub indices: k in [0, p), i in [0, n)
    
    def add_schoolbook(offset, j_offset):
        col = offset
        for i in range(n):
            for j in range(m1): # m1 is 2
                for k in range(p):
                    # Global U index: i * m + (j_offset + j)
                    U[i * m + (j_offset + j)][col] = 1
                    # Global V index: (j_offset + j) * p + k
                    V[(j_offset + j) * p + k][col] = 1
                    # Global W index: k * n + i
                    W[k * n + i][col] = 1
                    col += 1
        return col

    # Block 1: j_offset = 0
    next_col = add_schoolbook(0, 0)
    # Block 2: j_offset = 2
    next_col = add_schoolbook(next_col, 2)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
