import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin
    
    n, m, p = DIMENSIONS
    
    # Rank 26 decomposition
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal col
        U[u_idx][col] += coeff
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col += 1
        
    # 1. Strassen for top-left 2x2 (Rank 7)
    # M1
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    # M2
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1
    
    # M3
    U[0][col] = 1
    V[1][col] = 1; V[4][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    
    # M4
    U[4][col] = 1
    V[3][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # M5
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1
    
    # M6
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    
    # M7
    U[1][col] = 1; U[4][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[0][col] = 1
    col += 1
    
    # 2. Cross terms (Rank 19)
    # a b^T (2x1 * 1x2) -> Rank 4
    add_term(2, 6, 0, 1)
    add_term(2, 7, 3, 1)
    add_term(5, 6, 1, 1)
    add_term(5, 7, 4, 1)
    
    # A2 b (2x2 * 2x1) -> Rank 4
    add_term(0, 2, 6, 1)
    add_term(1, 5, 6, 1)
    add_term(3, 2, 7, 1)
    add_term(4, 5, 7, 1)
    
    # a b3 (2x1 * 1x1) -> Rank 2
    add_term(2, 8, 6, 1)
    add_term(5, 8, 7, 1)
    
    # a^T B2 (1x2 * 2x2) -> Rank 4
    add_term(6, 0, 2, 1)
    add_term(7, 3, 2, 1)
    add_term(6, 1, 5, 1)
    add_term(7, 4, 5, 1)
    
    # a3 b^T (1x1 * 1x2) -> Rank 2
    add_term(8, 6, 2, 1)
    add_term(8, 7, 5, 1)
    
    # a^T b (1x2 * 2x1) -> Rank 2
    add_term(6, 2, 8, 1)
    add_term(7, 5, 8, 1)
    
    # a3 b3 (1x1 * 1x1) -> Rank 1
    add_term(8, 8, 8, 1)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
