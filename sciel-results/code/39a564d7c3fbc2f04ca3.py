"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    # Decompose p=6 as 2*3: k -> (a,b) with a in {0,1}, b in {0,1,2}
    # Use Kronecker structure: rank = R1 * R2 = 2 * 3 = 6
    R1 = 2  # for the n=2 and a=2 parts
    R2 = 3  # for the m=3 and b=3 parts
    rank = R1 * R2  # 6
    
    # U: shape (n, rank) = (2, 6), indexed as U[i, (r,s)]
    # U[i, (r,s)] = delta(i, r)  (identity in r, ignore s)
    U = [[0] * rank for _ in range(n)]
    for i in range(n):
        for r in range(R1):
            for s in range(R2):
                col = r * R2 + s
                if i == r:
                    U[i][col] = 1
    
    # V: shape (m, rank) = (3, 6), indexed as V[j, (r,s)]
    # V[j, (r,s)] = delta(j, s)  (identity in s, ignore r)
    V = [[0] * rank for _ in range(m)]
    for j in range(m):
        for r in range(R1):
            for s in range(R2):
                col = r * R2 + s
                if j == s:
                    V[j][col] = 1
    
    # W: shape (p, rank) = (6, 6), indexed as W[k, (r,s)] where k = a*m2 + b, m2=2
    # W[(a,b), (r,s)] = W1[a,r] * W2[b,s] = 1 * 1 = 1 (all ones)
    W = [[0] * rank for _ in range(p)]
    for a in range(2):
        for b in range(3):
            k = a * 3 + b  # k = a*3 + b for p=6, splitting as 2x3
            for r in range(R1):
                for s in range(R2):
                    col = r * R2 + s
                    W[k][col] = 1
    
    # The interface expects U of shape (n*m, rank), V of shape (m*p, rank), W of shape (p*n, rank)
    # U[i*m+j, c] for schoolbook, but here our U is (n, rank)
    # We need to map: U_full[i*m+j, col] = U[i, col] (independent of j)
    U_full = [[0] * rank for _ in range(n * m)]
    for i in range(n):
        for j in range(m):
            for col in range(rank):
                U_full[i * m + j][col] = U[i][col]
    
    # V_full[j*p+k, col] = V[j, col] (independent of k)
    V_full = [[0] * rank for _ in range(m * p)]
    for j in range(m):
        for k in range(p):
            for col in range(rank):
                V_full[j * p + k][col] = V[j][col]
    
    # W_full[k*n+i, col] = W[k, col] (independent of i)
    W_full = [[0] * rank for _ in range(p * n)]
    for k in range(p):
        for i in range(n):
            for col in range(rank):
                W_full[k * n + i][col] = W[k][col]
    
    return {"U": U_full, "V": V_full, "W": W_full}
# EVOLVE-BLOCK-END
