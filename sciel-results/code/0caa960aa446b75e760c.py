"""Rewritten program for Matrix multiplication 2,2,13"""

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strategy: Decompose B into sum of p rank-1 matrices b_k e_k^T
    # C = A B = sum_k A b_k e_k^T
    # Each term A b_k e_k^T can be decomposed into n rank-1 tensor terms
    # Total rank = n * p
    rank = n * p
    
    U = [[0] * rank for _ in range(n * m)] # 4 x 26
    V = [[0] * rank for _ in range(m * p)] # 26 x 26
    W = [[0] * rank for _ in range(p * n)] # 26 x 26
    
    for k in range(p):
        for s in range(n):
            r = k * n + s
            
            # U[i*m + j] = delta(i, s)
            for i in range(n):
                for j in range(m):
                    U[i * m + j][r] = 1 if i == s else 0
            
            # V[j*p + k'] = delta(k', k)
            for j in range(m):
                for kp in range(p):
                    V[j * p + kp][r] = 1 if kp == k else 0
            
            # W[k'*n + i'] = delta(k', k) * delta(i', s)
            for kp in range(p):
                for ip in range(n):
                    W[kp * n + ip][r] = 1 if (kp == k and ip == s) else 0
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
