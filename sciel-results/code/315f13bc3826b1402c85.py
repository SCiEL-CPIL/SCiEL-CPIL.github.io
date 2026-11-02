import time

DIMENSIONS = [2, 2, 12]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=12
    # U size: n*m = 4
    # V size: m*p = 24
    # W size: p*n = 24
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # P1: (a00+a11)(b00+b11)
    # P2: (a10+a11)b00
    # P3: a00(b01-b11)
    # P4: a11(b10-b00)
    # P5: (a00+a01)b11
    # P6: (a10-a00)(b00+b01)
    # P7: (a01-a11)(b10+b11)
    
    # Output combinations:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # U indices:
    # u00=0, u01=1, u10=2, u11=3
    
    # For each of the 6 blocks of B (2x2)
    for block_idx in range(6):
        k0 = 2 * block_idx
        k1 = 2 * block_idx + 1
        
        # V indices for this block
        v00 = 0 * p + k0
        v01 = 0 * p + k1
        v10 = 1 * p + k0
        v11 = 1 * p + k1
        
        # W indices for this block
        # W index = k * n + i
        w00 = k0 * n + 0
        w01 = k1 * n + 0
        w10 = k0 * n + 1
        w11 = k1 * n + 1
        
        base_col = 7 * block_idx
        
        # P1
        col = base_col + 0
        U[0][col] = 1
        U[3][col] = 1
        V[v00][col] = 1
        V[v11][col] = 1
        W[w00][col] = 1
        W[w11][col] = 1
        
        # P2
        col = base_col + 1
        U[2][col] = 1
        U[3][col] = 1
        V[v00][col] = 1
        W[w10][col] = 1
        W[w11][col] = -1
        
        # P3
        col = base_col + 2
        U[0][col] = 1
        V[v01][col] = 1
        V[v11][col] = -1
        W[w01][col] = 1
        W[w11][col] = 1
        
        # P4
        col = base_col + 3
        U[3][col] = 1
        V[v10][col] = 1
        V[v00][col] = -1
        W[w00][col] = 1
        W[w10][col] = 1
        
        # P5
        col = base_col + 4
        U[0][col] = 1
        U[1][col] = 1
        V[v11][col] = 1
        W[w00][col] = -1
        W[w01][col] = 1
        
        # P6
        col = base_col + 5
        U[2][col] = 1
        U[0][col] = -1
        V[v00][col] = 1
        V[v01][col] = 1
        W[w11][col] = 1
        
        # P7
        col = base_col + 6
        U[1][col] = 1
        U[3][col] = -1
        V[v10][col] = 1
        V[v11][col] = 1
        W[w00][col] = 1

    return {"U": U, "V": V, "W": W}
