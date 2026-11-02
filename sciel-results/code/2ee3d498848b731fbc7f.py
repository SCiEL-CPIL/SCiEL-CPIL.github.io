import time
_start_time = time.time()
_DEADLINE = _start_time + 145.0

def construct():
    n, m, p = 2, 2, 15
    
    # Rank 53 decomposition:
    # 7 Strassen blocks for 14 columns (rank 7 each)
    # 1 Schoolbook block for 1 column (rank 4)
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(base_r, k0, k1):
        r1 = base_r
        r2 = base_r + 1
        r3 = base_r + 2
        r4 = base_r + 3
        r5 = base_r + 4
        r6 = base_r + 5
        r7 = base_r + 6
        
        # U matrix (A coefficients)
        U[0][r1] = 1; U[3][r1] = 1
        U[2][r2] = 1; U[3][r2] = 1
        U[0][r3] = 1
        U[3][r4] = 1
        U[0][r5] = 1; U[1][r5] = 1
        U[2][r6] = 1; U[0][r6] = -1
        U[1][r7] = 1; U[3][r7] = -1
        
        # V matrix (B coefficients)
        # Indices in V are m*p = 30. Row index = i*p + j
        # k0, k1 are column indices in B
        V[0 * p + k0][r1] = 1; V[1 * p + k1][r1] = 1
        V[0 * p + k0][r2] = 1
        V[0 * p + k1][r3] = 1; V[1 * p + k1][r3] = -1
        V[1 * p + k0][r4] = 1; V[0 * p + k0][r4] = -1
        V[1 * p + k1][r5] = 1
        V[0 * p + k0][r6] = 1; V[0 * p + k1][r6] = 1
        V[1 * p + k0][r7] = 1; V[1 * p + k1][r7] = 1
        
        # W matrix (C coefficients)
        # Indices in W are p*n = 30. Row index = k*n + i
        # k0, k1 are column indices in C
        W[2 * k0][r1] = 1; W[2 * k0][r4] = 1; W[2 * k0][r5] = -1; W[2 * k0][r7] = 1
        W[2 * k0 + 1][r2] = 1; W[2 * k0 + 1][r4] = 1
        W[2 * k1][r3] = 1; W[2 * k1][r5] = 1
        W[2 * k1 + 1][r1] = 1; W[2 * k1 + 1][r2] = -1; W[2 * k1 + 1][r3] = 1; W[2 * k1 + 1][r6] = 1

    # Add 7 Strassen blocks for columns 0-13
    for i in range(7):
        add_strassen_block(7 * i, 2 * i, 2 * i + 1)
        
    # Add 1 Schoolbook block for column 14
    base_r = 49
    k = 14
    U[0][base_r] = 1; V[0 * p + k][base_r] = 1; W[2 * k][base_r] = 1
    U[1][base_r+1] = 1; V[1 * p + k][base_r+1] = 1; W[2 * k][base_r+1] = 1
    U[2][base_r+2] = 1; V[0 * p + k][base_r+2] = 1; W[2 * k + 1][base_r+2] = 1
    U[3][base_r+3] = 1; V[1 * p + k][base_r+3] = 1; W[2 * k + 1][base_r+3] = 1
    
    return {"U": U, "V": V, "W": W}
