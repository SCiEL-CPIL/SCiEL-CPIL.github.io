"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's algorithm for 2x2 matrix multiplication
    # A = [[a, b], [c, d]] -> indices 0,1,2,3 in U
    # B = [[e, f], [g, h]] -> indices 0,1,2,3 in V
    # C = [[c00, c01], [c10, c11]] -> indices k*n+i in W
    # n=2, so indices:
    # c00: k=0, i=0 -> 0
    # c10: k=0, i=1 -> 1
    # c01: k=1, i=0 -> 2
    # c11: k=1, i=1 -> 3
    
    # M1 = (a+d)(e+h)
    U[0][0] = 1  # a
    U[3][0] = 1  # d
    V[0][0] = 1  # e
    V[3][0] = 1  # h
    
    # M2 = (c+d)e
    U[2][1] = 1  # c
    U[3][1] = 1  # d
    V[0][1] = 1  # e
    
    # M3 = a(f-h)
    U[0][2] = 1  # a
    V[1][2] = 1  # f
    V[3][2] = -1 # h
    
    # M4 = d(g-e)
    U[3][3] = 1  # d
    V[2][3] = 1  # g
    V[0][3] = -1 # e
    
    # M5 = (a+b)h
    U[0][4] = 1  # a
    U[1][4] = 1  # b
    V[3][4] = 1  # h
    
    # M6 = (c-a)(e+f)
    U[2][5] = 1  # c
    U[0][5] = -1 # a
    V[0][5] = 1  # e
    V[1][5] = 1  # f
    
    # M7 = (b-d)(g+h)
    U[1][6] = 1  # b
    U[3][6] = -1 # d
    V[2][6] = 1  # g
    V[3][6] = 1  # h
    
    # W coefficients for C = A*B
    # C00 = M1 - M4 + M5 + M7
    W[0][0] = 1  # M1
    W[0][3] = -1 # M4
    W[0][4] = 1  # M5
    W[0][6] = 1  # M7
    
    # C10 = M1 + M2
    W[1][0] = 1  # M1
    W[1][1] = 1  # M2
    
    # C01 = M3 + M4
    W[2][2] = 1  # M3
    W[2][3] = 1  # M4
    
    # C11 = M4 - M2 + M3 + M6
    W[3][3] = 1  # M4
    W[3][1] = -1 # M2
    W[3][2] = 1  # M3
    W[3][5] = 1  # M6
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
