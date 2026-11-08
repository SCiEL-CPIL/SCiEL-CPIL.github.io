"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    
    # Naive decomposition rank: n * m * p
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]   # (n*m) x rank
    V = [[0] * rank for _ in range(m * p)]   # (m*p) x rank
    W = [[0] * rank for _ in range(p * n)]   # (p*n) x rank
    
    # Iterate over the contraction dimension j (0 to m-1)
    # For each j, iterate over i (0 to n-1) and k (0 to p-1)
    for j in range(m):
        for i in range(n):
            for k in range(p):
                # Calculate the column index in the factor matrices
                # We map (j, i, k) to a unique rank index r
                # r = j * (n * p) + i * p + k
                r = j * (n * p) + i * p + k
                
                # U corresponds to A[i, j]
                # Flat index in U: i * m + j
                u_idx = i * m + j
                U[u_idx][r] = 1
                
                # V corresponds to B[j, k]
                # Flat index in V: j * p + k
                v_idx = j * p + k
                V[v_idx][r] = 1
                
                # W corresponds to C[i, k]
                # Convention: W[k * n + i, r]
                w_idx = k * n + i
                W[w_idx][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
