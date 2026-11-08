"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    
    # Naive decomposition: Rank = n * m * p = 2 * 3 * 6 = 36
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]   # 6 x 36
    V = [[0] * rank for _ in range(m * p)]   # 18 x 36
    W = [[0] * rank for _ in range(p * n)]   # 12 x 36
    
    # Indices:
    # A: (i, j) -> i*m + j, i in 0..n-1, j in 0..m-1
    # B: (j, k) -> j*p + k, j in 0..m-1, k in 0..p-1
    # C: (i, k) -> k*n + i (Trace convention W[k*n+i, r])
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Term for A[i,j] * B[j,k] contributing to C[i,k]
                
                # U: A[i,j]
                u_idx = i * m + j
                U[u_idx][idx] = 1
                
                # V: B[j,k]
                v_idx = j * p + k
                V[v_idx][idx] = 1
                
                # W: C[i,k]
                # Convention: W[k*n + i, r]
                w_idx = k * n + i
                W[w_idx][idx] = 1
                
                idx += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
