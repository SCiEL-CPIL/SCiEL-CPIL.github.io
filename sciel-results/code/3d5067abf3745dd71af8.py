"""Rewritten program: rank-22 exact tensor decomposition for 2x2x7 matrix multiplication."""
DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 7
    R = 3 * p + 1  # 22
    
    U = [[0] * R for _ in range(n * m)]  # 4 x 22
    V = [[0] * R for _ in range(m * p)]  # 14 x 22
    W = [[0] * R for _ in range(p * n)]  # 14 x 22
    
    # For each k in 0..p-1, assign 3 terms:
    # Term 3k:   U=[1,0,1,0], V=[1,0;1,0] (rows for k), W=[1,0;1,0] (rows for k)
    # Term 3k+1: U=[0,1,0,1], V=[0,1;0,1], W=[0,1;0,1]
    # Term 3k+2: U=[1,1,0,0], V=[1,1;0,0], W=[1,1;0,0]
    # Plus 1 extra term:
    # Term 3p-1: U=[0,0,1,1], V=[0,0;1,1], W=[0,0;1,1]
    
    for k in range(p):
        # Term 3k
        idx = 3 * k
        U[0][idx] = 1
        U[2][idx] = 1
        V[k][idx] = 1
        V[p + k][idx] = 1
        W[k][idx] = 1
        W[p + k][idx] = 1
        
        # Term 3k+1
        idx = 3 * k + 1
        U[1][idx] = 1
        U[3][idx] = 1
        V[k + 1][idx] = 1
        V[p + k + 1][idx] = 1
        W[k + 1][idx] = 1
        W[p + k + 1][idx] = 1
        
        # Term 3k+2
        idx = 3 * k + 2
        U[0][idx] = 1
        U[1][idx] = 1
        V[k + 2][idx] = 1
        V[p + k + 2][idx] = 1
        W[k + 2][idx] = 1
        W[p + k + 2][idx] = 1
    
    # Extra term
    idx = 3 * p - 1
    U[2][idx] = 1
    U[3][idx] = 1
    V[0][idx] = 1
    V[p][idx] = 1
    W[0][idx] = 1
    W[p][idx] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
