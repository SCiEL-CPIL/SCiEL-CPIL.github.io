"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=8
    # We decompose into two 4x4x4 multiplications.
    # Rank for each 4x4x4 is 16 (schoolbook).
    # Total rank = 32.
    
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Left half: k in 0..3
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(4):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    # Right half: k in 4..7
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(4, 8):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col + 16] = 1
                V[v_idx][col + 16] = 1
                W[w_idx][col + 16] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
