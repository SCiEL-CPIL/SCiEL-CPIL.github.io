"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank estimate: 54
    rank = 54
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a schoolbook block
    def add_block(i_start, i_end, j_start, j_end, k_start, k_end):
        nonlocal col
        for i in range(i_start, i_end + 1):
            for j in range(j_start, j_end + 1):
                for k in range(k_start, k_end + 1):
                    # U index: i*m + j
                    # V index: j*p + k
                    # W index: k*n + i
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

    # Block 1: i=1..3, j=1..3, k=1..3
    add_block(1, 3, 1, 3, 1, 3)
    
    # Block 2: i=1..3, j=1..3, k=4..6
    add_block(1, 3, 1, 3, 4, 6)
    
    # Block 3: i=1..3, j=4..5, k=1..6
    add_block(1, 3, 4, 5, 1, 6)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
