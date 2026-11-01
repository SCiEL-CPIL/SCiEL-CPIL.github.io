"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 40 decomposition for 2x2 by 2x13
    # Base Strassen for p=2 (cols 0,1) has rank 7.
    # Each additional column k>=2 adds 3 multiplications.
    # Total rank = 7 + 3 * (13 - 2) = 7 + 33 = 40.
    
    rank = 40
    U = [[0] * rank for _ in range(n * m)] # 4 x 40
    V = [[0] * rank for _ in range(m * p)] # 26 x 40
    W = [[0] * rank for _ in range(p * n)] # 26 x 40
    
    # Strassen factors for cols 0,1
    # U indices: 0=a, 1=b, 2=c, 3=d
    # V indices: j*p + k. p=13.
    # k=0: 0, 13
    # k=1: 1, 14
    # W indices: k*n + i. n=2.
    # k=0: 0, 1
    # k=1: 2, 3
    
    # M1: (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[1][0] = 1 # e, f (k=0,1 for j=0)
    W[0][0] = 1; W[3][0] = 1 # C00, C11
    
    # M2: (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1 # e (k=0 for j=0)
    W[2][1] = 1; W[3][1] = -1 # C10, C11
    
    # M3: a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[14][2] = -1 # f (k=1, j=0), h (k=1, j=1)
    W[1][2] = 1; W[3][2] = 1 # C01, C11
    
    # M4: d(g-f)
    U[3][3] = 1
    V[13][3] = 1; V[1][3] = -1 # g (k=0, j=1), f (k=1, j=0)
    W[0][3] = 1; W[2][3] = 1 # C00, C10
    
    # M5: (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[14][4] = 1 # h (k=1, j=1)
    W[0][4] = -1; W[1][4] = 1 # C00, C01
    
    # M6: (c-a)(e+f)
    U[0][5] = -1; U[2][5] = 1
    V[0][5] = 1; V[1][5] = 1 # e, f
    W[3][5] = 1 # C11
    
    # M7: (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[13][6] = 1; V[14][6] = 1 # g, h
    W[0][6] = 1 # C00
    
    # Extension for k = 2 to 12
    for k in range(2, p):
        base_r = 7 + 3 * (k - 2)
        
        # M_k1: a * B_0k
        U[0][base_r] = 1
        V[k][base_r] = 1 # j=0, k
        W[2*k][base_r] = 1 # C_0k
        W[2*k+1][base_r] = 1 # C_1k
        
        # M_k2: b * B_1k
        U[1][base_r + 1] = 1
        V[13 + k][base_r + 1] = 1 # j=1, k
        W[2*k][base_r + 1] = 1 # C_0k
        W[2*k+1][base_r + 1] = 1 # C_1k
        
        # M_k3: (c-a)B_0k + (d-b)B_1k
        U[0][base_r + 2] = -1
        U[1][base_r + 2] = -1
        U[2][base_r + 2] = 1
        U[3][base_r + 2] = 1
        V[k][base_r + 2] = 1
        V[13 + k][base_r + 2] = 1
        W[2*k+1][base_r + 2] = 1 # C_1k only
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
