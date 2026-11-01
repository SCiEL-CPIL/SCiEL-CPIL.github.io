<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    # Try Strassen-based rank-49 decomposition first
    result_49 = _construct_strassen_49(n, m, p)
    if result_49 is not None:
        return result_49
    
    # Fallback: schoolbook rank-56
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
=======
# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank-m decomposition
    # C[i,k] = sum_j A[i,j] B[j,k]
    # We decompose into m terms, one for each j
    # Term r (corresponding to j=r):
    # U[i*m+j, r] = 1 if j==r else 0
    # V[j*p+k, r] = 1 if j==r else 0
    # W[k*n+i, r] = 1
    
    rank = m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(m):
        # U: indices i*m + j
        for i in range(n):
            U[i * m + r][r] = 1
            
        # V: indices j*p + k
        for k in range(p):
            V[r * p + k][r] = 1
            
        # W: indices k*n + i
        for k in range(p):
            for i in range(n):
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE
