import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # Part 1: 2x2x10 using Strassen on 2x2x2 blocks (5 blocks * 7 rank) = 35
    # Part 2: 2x1x10 using Schoolbook (2 * 1 * 10) = 20
    # Total Rank = 55
    rank = 55
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 2: Schoolbook for j = 2 (0-indexed), i in 0..1, k in 0..9
    # U[i*m + j], V[j*p + k], W[k*n + i]
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 1: Strassen for j in 0,1
    # Split k into pairs
    for block_idx in range(p // 2):
        k1 = block_idx * 2
        k2 = block_idx * 2 + 1
        
        # Global indices for this 2x2x2 block
        # U indices: i in 0,1; j in 0,1
        u00 = 0 * m + 0
        u01 = 0 * m + 1
        u10 = 1 * m + 0
        u11 = 1 * m + 1
        
        # V indices: j in 0,1; k in k1,k2
        v00 = 0 * p + k1
        v01 = 0 * p + k2
        v10 = 1 * p + k1
        v11 = 1 * p + k2
        
        # W indices: k in k1,k2; i in 0,1
        w00 = k1 * n + 0
        w01 = k1 * n + 1
        w10 = k2 * n + 0
        w11 = k2 * n + 1
        
        # Strassen Steps
        # M1 = (A00 + A11)(B00 + B11) -> C00, C11
        U[u00][col] += 1
        U[u11][col] += 1
        V[v00][col] += 1
        V[v11][col] += 1
        W[w00][col] += 1
        W[w11][col] += 1
        col += 1
        
        # M2 = (A10 + A11)B00 -> C10, C11
        U[u10][col] += 1
        U[u11][col] += 1
        V[v00][col] += 1
        W[w10][col] += 1
        W[w11][col] += -1 # C11 gets -M2
        col += 1
        
        # M3 = A00(B01 - B11) -> C01, C11
        U[u00][col] += 1
        V[v01][col] += 1
        V[v11][col] += -1
        W[w01][col] += 1
        W[w11][col] += 1 # C11 gets +M3
        col += 1
        
        # M4 = A11(B10 - B00) -> C00, C10
        U[u11][col] += 1
        V[v10][col] += 1
        V[v00][col] += -1
        W[w00][col] += 1 # C00 gets +M4
        W[w10][col] += 1
        col += 1
        
        # M5 = (A00 + A01)B11 -> C00, C01
        U[u00][col] += 1
        U[u01][col] += 1
        V[v11][col] += 1
        W[w00][col] += -1 # C00 gets -M5
        W[w01][col] += 1
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01) -> C11
        U[u10][col] += 1
        U[u00][col] += -1
        V[v00][col] += 1
        V[v01][col] += 1
        W[w11][col] += 1
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11) -> C00
        U[u01][col] += 1
        U[u11][col] += -1
        V[v10][col] += 1
        V[v11][col] += 1
        W[w00][col] += -1 # C00 gets -M7
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
