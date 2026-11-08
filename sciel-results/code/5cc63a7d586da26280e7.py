"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    
    # Naive decomposition: rank = n*m*p = 2*3*6 = 36
    # Each term corresponds to a specific (i, j, k) triple
    # U[i*m + j, term] = 1 if this term uses A[i,j]
    # V[j*p + k, term] = 1 if this term uses B[j,k]
    # W[k*n + i, term] = 1 if this term contributes to C[i,k]
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]   # 6 x 36
    V = [[0] * rank for _ in range(m * p)]   # 18 x 36
    W = [[0] * rank for _ in range(p * n)]   # 12 x 36
    
    term_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                u_idx = i * m + j
                # V index: j*p + k
                v_idx = j * p + k
                # W index: k*n + i
                w_idx = k * n + i
                
                U[u_idx][term_idx] = 1
                V[v_idx][term_idx] = 1
                W[w_idx][term_idx] = 1
                term_idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
