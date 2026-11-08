"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    
    # Rank is n * m * p
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]   # 6 x 36
    V = [[0] * rank for _ in range(m * p)]   # 18 x 36
    W = [[0] * rank for _ in range(p * n)]   # 12 x 36
    
    # Iterate over all combinations of i, j, k
    # i: row of A, col of C (0..n-1)
    # j: col of A, row of B (0..m-1)
    # k: col of B, col of C (0..p-1)
    
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                u_idx = i * m + j
                # V index: j*p + k
                v_idx = j * p + k
                # W index: k*n + i (Trace convention W[k*n+i, r])
                w_idx = k * n + i
                
                U[u_idx][t] = 1
                V[v_idx][t] = 1
                W[w_idx][t] = 1
                
                t += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
