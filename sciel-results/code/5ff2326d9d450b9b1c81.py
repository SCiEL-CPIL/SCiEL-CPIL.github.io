"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Block-wise decomposition: split middle dim 5 = 3 + 2.
    C = A1*B1 + A2*B2 where A1(3x3), B1(3x7), A2(3x2), B2(2x7).
    
    For each sub-problem, use the naive per-(i,j,k) decomposition.
    Total rank = 3*3*7 + 3*2*7 = 63 + 42 = 105.
    
    This is correct but not optimal. We can improve by using
    Strassen-like tricks on the 2x2 sub-block.
    
    For now, return the correct 105-rank decomposition.
    """
    n, m, p = DIMENSIONS  # 3, 5, 7
    m1, m2 = 3, 2  # split of middle dimension
    
    # Build decomposition for A1(3xm1) * B1(m1xp) and A2(3xm2) * B2(m2xp)
    U_list = []
    V_list = []
    W_list = []
    
    # Sub-problem 1: 3 x 3 x 7
    for i in range(n):
        for j in range(m1):
            for k in range(p):
                # U index: i*m + j (since j is in [0, m1))
                # V index: j*p + k
                # W index: k*n + i
                U_list.append([1])
                V_list.append([1])
                W_list.append([1])
    
    # Sub-problem 2: 3 x 2 x 7
    for i in range(n):
        for j in range(m1, m):
            for k in range(p):
                U_list.append([1])
                V_list.append([1])
                W_list.append([1])
    
    rank = len(U_list)
    
    # Now build the actual matrices with correct indices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Sub-problem 1: j in [0, m1)
    for i in range(n):
        for j in range(m1):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Sub-problem 2: j in [m1, m)
    for i in range(n):
        for j in range(m1, m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
