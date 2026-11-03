import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    rank_part1 = 1 * m * p  # 32
    rank_part2 = 2 * (p // 2) * 7 # 56
    total_rank = rank_part1 + rank_part2 # 88
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    col = 0
    
    # Part 1: Row 0 of A (1x4) times B (4x8)
    for k in range(p):
        for j in range(m):
            U[0 * m + j, col] = 1.0
            V[j * p + k, col] = 1.0
            W[k * n + 0, col] = 1.0
            col += 1
            
    # Part 2: Rows 1,2 of A (2x4) times B (4x8)
    def add_strassen_block(I, J, K, col_start):
        i0, i1 = I
        j0, j1 = J
        k0, k1 = K
        
        c = col_start
        
        # P1 = (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
        U[i0*m + j0, c] = 1.0
        U[i1*m + j1, c] = 1.0
        V[j0*p + k0, c] = 1.0
        V[j1*p + k1, c] = 1.0
        W[k0*n + i0, c] = 1.0
        W[k1*n + i1, c] = 1.0
        c += 1
        
        # P2 = (A[i1,j0] + A[i1,j1]) * B[j0,k0]
        U[i1*m + j0, c] = 1.0
        U[i1*m + j1, c] = 1.0
        V[j0*p + k0, c] = 1.0
        W[k0*n + i1, c] = 1.0
        c += 1
        
        # P3 = A[i0,j0] * (B[j0,k1] - B[j1,k1])
        U[i0*m + j0, c] = 1.0
        V[j0*p + k1, c] = 1.0
        V[j1*p + k1, c] = -1.0
        W[k1*n + i0, c] = 1.0
        c += 1
        
        # P4 = A[i1,j1] * (B[j1,k0] - B[j0,k0])
        U[i1*m + j1, c] = 1.0
        V[j1*p + k0, c] = 1.0
        V[j0*p + k0, c] = -1.0
        W[k0*n + i0, c] = 1.0
        W[k0*n + i1, c] = 1.0
        c += 1
        
        # P5 = (A[i0,j0] + A[i0,j1]) * B[j1,k1]
        U[i0*m + j0, c] = 1.0
        U[i0*m + j1, c] = 1.0
        V[j1*p + k1, c] = 1.0
        W[k0*n + i0, c] = -1.0
        W[k1*n + i0, c] = 1.0
        c += 1
        
        # P6 = (A[i1,j0] - A[i0,j0]) * (B[j0,k0] + B[j0,k1])
        U[i1*m + j0, c] = 1.0
        U[i0*m + j0, c] = -1.0
        V[j0*p + k0, c] = 1.0
        V[j0*p + k1, c] = 1.0
        W[k1*n + i1, c] = 1.0
        c += 1
        
        # P7 = (A[i0,j1] - A[i1,j1]) * (B[j1,k0] + B[j1,k1])
        U[i0*m + j1, c] = 1.0
        U[i1*m + j1, c] = -1.0
        V[j1*p + k0, c] = 1.0
        V[j1*p + k1, c] = 1.0
        W[k0*n + i0, c] = 1.0
        c += 1
        
        return c

    # AL * BT
    for q in range(4):
        k0 = 2*q
        k1 = 2*q + 1
        col = add_strassen_block([1, 2], [0, 1], [k0, k1], col)
        
    # AR * BB
    for q in range(4):
        k0 = 2*q
        k1 = 2*q + 1
        col = add_strassen_block([1, 2], [2, 3], [k0, k1], col)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
