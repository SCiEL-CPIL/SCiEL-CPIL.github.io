"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Dimensions for sub-problems
    n1, m1, p1 = 2, 2, 2
    n2, m2, p2 = 1, 2, 4
    
    # Rank of sub-problems
    rank1 = 7
    rank2 = 8
    rank = rank1 * rank2
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 matrices
    # U1: (n1*m1) x rank1 = 4 x 7
    U1 = [
        [1, 0, 1, 0, 1, -1, 0], # a00
        [0, 0, 0, 0, 1, 0, 1],  # a01
        [0, 1, 0, 0, 0, 1, 0],  # a10
        [1, 1, 0, 1, 0, 0, -1]  # a11
    ]
    
    # V1: (m1*p1) x rank1 = 4 x 7
    V1 = [
        [1, 1, 0, -1, 0, 1, 0], # b00
        [0, 0, 1, 0, 0, 1, 0],  # b01
        [0, 0, 0, 1, 0, 0, 1],  # b10
        [1, 0, -1, 0, 1, 0, 1]  # b11
    ]
    
    # W1: (p1*n1) x rank1 = 4 x 7
    W1 = [
        [1, 0, 0, 1, -1, 0, 1], # c00
        [0, 0, 1, 0, 1, 0, 0],  # c01
        [0, 1, 0, 1, 0, 0, 0],  # c10
        [1, -1, 1, 0, 0, 1, 0]  # c11
    ]
    
    # Schoolbook 1x2x4 matrices
    # U2: (n2*m2) x rank2 = 2 x 8
    U2 = [
        [1, 0, 1, 0, 1, 0, 1, 0], # a00
        [0, 1, 0, 1, 0, 1, 0, 1]  # a01
    ]
    
    # V2: (m2*p2) x rank2 = 8 x 8
    V2 = [
        [1, 0, 0, 0, 0, 0, 0, 0], # b00
        [0, 0, 1, 0, 0, 0, 0, 0], # b01
        [0, 0, 0, 0, 1, 0, 0, 0], # b02
        [0, 0, 0, 0, 0, 0, 1, 0], # b03
        [0, 1, 0, 1, 0, 1, 0, 1], # b10
        [0, 0, 0, 0, 0, 0, 0, 0], # placeholder, filled below
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ]
    # Correct V2 construction
    V2 = [[0]*rank2 for _ in range(m2*p2)]
    for k in range(p2):
        V2[k][2*k] = 1       # b0k
        V2[m2+k][2*k+1] = 1  # b1k
        
    # W2: (p2*n2) x rank2 = 4 x 8
    W2 = [[0]*rank2 for _ in range(p2*n2)]
    for k in range(p2):
        W2[k][2*k] = 1
        W2[k][2*k+1] = 1
        
    # Combine
    for c in range(rank1):
        for d in range(rank2):
            col = c * rank2 + d
            for i in range(n):
                i1 = i // n2
                i2 = i % n2
                for j in range(m):
                    j1 = j // m2
                    j2 = j % m2
                    # U[i*m+j, col]
                    u_val = U1[i1*m1+j1][c] * U2[i2*m2+j2][d]
                    U[i*m+j][col] = u_val
                    
                    for k in range(p):
                        k1 = k // p2
                        k2 = k % p2
                        # V[j*p+k, col]
                        v_val = V1[j1*p1+k1][c] * V2[j2*p2+k2][d]
                        V[j*p+k][col] = v_val
                        
                        # W[k*n+i, col]
                        w_val = W1[k1*n1+i1][c] * W2[k2*n2+i2][d]
                        W[k*n+i][col] = w_val
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
