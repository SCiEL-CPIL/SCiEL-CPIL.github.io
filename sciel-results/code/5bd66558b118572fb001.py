# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Direct construction of rank 10 decomposition
    # Based on two independent rank-5 decompositions of 1x5x5 matrix multiplication
    rank = 10
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        i = r // 5
        k = r % 5
        
        # U[i*m + j][r] = 1 if j == k
        U[i * m + k][r] = 1
        
        # V[j'*p + k'][r] = 1 if j' == k and k' == k
        V[k * p + k][r] = 1
        
        # W[k''*n + i''][r] = 1 if i'' == i and k'' == k
        W[k * n + i][r] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
