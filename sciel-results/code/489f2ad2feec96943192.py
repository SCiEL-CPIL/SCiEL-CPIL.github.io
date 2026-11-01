"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 8
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for first two columns (k=0, 1)
    # M1 = A[1,1] * (B[0,0] - B[1,0])
    U[1*m + 1][0] = 1
    V[0*p + 0][0] = 1
    V[1*p + 0][0] = -1
    W[0*n + 0][0] = -1 # C[0,0]
    W[1*n + 0][0] = 0  # C[1,0]
    W[2*n + 0][0] = 1  # C[0,1]
    W[3*n + 0][0] = 0  # C[1,1]
    
    # M2 = (A[0,0] + A[1,0]) * B[1,0]
    U[0*m + 0][1] = 1
    U[1*m + 0][1] = 1
    V[1*p + 0][1] = 1
    W[0*n + 0][1] = 0
    W[1*n + 0][1] = 1
    W[2*n + 0][1] = 0
    W[3*n + 0][1] = -1
    
    # M3 = A[0,1] * (B[1,0] + B[1,1])
    U[0*m + 1][2] = 1
    V[1*p + 0][2] = 1
    V[1*p + 1][2] = 1
    W[0*n + 0][2] = 0
    W[1*n + 0][2] = 0
    W[2*n + 0][2] = 1
    W[3*n + 0][2] = 1
    
    # M4 = (A[1,1] - A[0,1]) * B[0,1]
    U[1*m + 1][3] = 1
    U[0*m + 1][3] = -1
    V[0*p + 1][3] = 1
    W[0*n + 0][3] = 1
    W[1*n + 0][3] = 0
    W[2*n + 0][3] = 0
    W[3*n + 0][3] = 0
    
    # M5 = (A[0,0] + A[0,1]) * B[0,0]
    U[0*m + 0][4] = 1
    U[0*m + 1][4] = 1
    V[0*p + 0][4] = 1
    W[0*n + 0][4] = 1
    W[1*n + 0][4] = 0
    W[2*n + 0][4] = 0
    W[3*n + 0][4] = 1
    
    # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    U[1*m + 0][5] = 1
    U[0*m + 0][5] = -1
    V[0*p + 0][5] = 1
    V[0*p + 1][5] = 1
    W[0*n + 0][5] = 0
    W[1*n + 0][5] = 0
    W[2*n + 0][5] = 0
    W[3*n + 0][5] = 1
    
    # M7 = (A[1,1] - A[0,1]) * (B[0,1] - B[1,1])
    U[1*m + 1][6] = 1
    U[0*m + 1][6] = -1
    V[0*p + 1][6] = 1
    V[1*p + 1][6] = -1
    W[0*n + 0][6] = -1
    W[1*n + 0][6] = 1
    W[2*n + 0][6] = 0
    W[3*n + 0][6] = -1
    
    # M8 for third column (k=2)
    # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2]
    # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2]
    # Rank 1 decomposition: U=1, V=1 (for k=2), W=1 (for i=0,1)
    for i in range(n):
        for j in range(m):
            U[i*m + j][7] = 1
    for j in range(m):
        V[j*p + 2][7] = 1
    W[2*n + 0][7] = 1 # C[0,2]
    W[2*n + 1][7] = 1 # C[1,2]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
