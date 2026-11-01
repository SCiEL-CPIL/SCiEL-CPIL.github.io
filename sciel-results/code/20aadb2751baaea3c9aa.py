import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=14
    # U: (n*m, rank) = (4, rank)
    # V: (m*p, rank) = (28, rank)
    # W: (p*n, rank) = (28, rank)
    
    # Strassen on 2x2 blocks
    # Number of blocks is p // 2 = 7
    num_blocks = p // 2
    rank_per_block = 7
    total_rank = num_blocks * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for b in range(num_blocks):
        # Indices for B block b
        # B is m x p (2 x 14)
        # Block b consists of columns 2b and 2b+1
        # V index for B[j, k] is j * p + k
        v_e = 0 * p + (2 * b)       # B[0, 2b]
        v_f = 0 * p + (2 * b + 1)   # B[0, 2b+1]
        v_g = 1 * p + (2 * b)       # B[1, 2b]
        v_h = 1 * p + (2 * b + 1)   # B[1, 2b+1]
        
        # Indices for C block b
        # C is n x p (2 x 14)
        # W index for C[i, k] is k * n + i
        w_11 = (2 * b) * n + 0      # C[0, 2b]
        w_21 = (2 * b) * n + 1      # C[1, 2b]
        w_12 = (2 * b + 1) * n + 0  # C[0, 2b+1]
        w_22 = (2 * b + 1) * n + 1  # C[1, 2b+1]
        
        # U indices for A (same for all blocks)
        # A is n x m (2 x 2)
        # U index for A[i, j] is i * m + j
        u_a = 0 * m + 0  # A[0,0]
        u_b = 0 * m + 1  # A[0,1]
        u_c = 1 * m + 0  # A[1,0]
        u_d = 1 * m + 1  # A[1,1]
        
        # Strassen terms
        # m1 = (a+d)(e+h)
        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1
        W[w_11][col] += 1
        W[w_22][col] += 1
        col += 1
        
        # m2 = (c+d)e
        U[u_c][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        W[w_21][col] += 1
        W[w_22][col] += -1
        col += 1
        
        # m3 = a(f-h)
        U[u_a][col] += 1
        V[v_f][col] += 1
        V[v_h][col] += -1
        W[w_12][col] += 1
        W[w_22][col] += 1
        col += 1
        
        # m4 = d(g-e)
        U[u_d][col] += 1
        V[v_g][col] += 1
        V[v_e][col] += -1
        W[w_11][col] += 1
        W[w_21][col] += 1
        col += 1
        
        # m5 = (a+b)h
        U[u_a][col] += 1
        U[u_b][col] += 1
        V[v_h][col] += 1
        W[w_11][col] += -1
        W[w_12][col] += 1
        col += 1
        
        # m6 = (c-a)(e+f)
        U[u_c][col] += 1
        U[u_a][col] += -1
        V[v_e][col] += 1
        V[v_f][col] += 1
        W[w_22][col] += 1
        col += 1
        
        # m7 = (b-d)(g+h)
        U[u_b][col] += 1
        U[u_d][col] += -1
        V[v_g][col] += 1
        V[v_h][col] += 1
        W[w_11][col] += 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
