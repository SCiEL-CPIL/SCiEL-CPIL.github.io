import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    # Split p into 5 blocks of size 2
    num_blocks = p // 2
    block_size = 2
    
    # Total rank is 5 * (7 + 4) = 55
    rank = num_blocks * (7 + 4)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We define the terms (U_idx, V_idx, W_idx, coeff)
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # For each block
    for b in range(num_blocks):
        # B columns are 2*b and 2*b+1
        k1 = 2 * b
        k2 = 2 * b + 1
        
        # --- Strassen Part (A[:, 0:2] * B[0:2, :]) ---
        # A indices: j=0, j=1
        # B indices: j=0, j=1 (rows of B)
        
        # M1
        U[0*m + 0][col] = 1
        U[1*m + 1][col] = 1
        V[0*p + k1][col] = 1
        V[1*p + k1][col] = 1
        V[0*p + k2][col] = 1
        V[1*p + k2][col] = 1
        W[k1*n + 0][col] = 1
        W[k2*n + 0][col] = 1
        W[k1*n + 1][col] = 1
        W[k2*n + 1][col] = 1
        col += 1
        
        # M2
        U[1*m + 0][col] = 1
        U[1*m + 1][col] = 1
        V[0*p + k1][col] = 1
        W[k1*n + 1][col] = 1
        W[k2*n + 1][col] = 1
        col += 1
        
        # M3
        U[0*m + 0][col] = 1
        V[0*p + k2][col] = 1
        V[1*p + k2][col] = -1
        W[k2*n + 0][col] = 1
        col += 1
        
        # M4
        U[1*m + 1][col] = 1
        V[1*p + k1][col] = 1
        V[0*p + k1][col] = -1
        W[k1*n + 1][col] = 1
        col += 1
        
        # M5
        U[0*m + 0][col] = 1
        U[0*m + 1][col] = 1
        V[1*p + k2][col] = 1
        W[k2*n + 0][col] = -1
        W[k2*n + 1][col] = 1
        col += 1
        
        # M6
        U[1*m + 0][col] = 1
        U[0*m + 0][col] = -1
        V[0*p + k1][col] = 1
        V[0*p + k2][col] = 1
        W[k1*n + 1][col] = 1
        W[k2*n + 1][col] = 1
        col += 1
        
        # M7
        U[0*m + 1][col] = 1
        U[1*m + 1][col] = -1
        V[1*p + k1][col] = 1
        V[1*p + k2][col] = 1
        W[k1*n + 0][col] = 1
        W[k2*n + 0][col] = -1
        col += 1
        
        # --- Schoolbook Part (A[:, 2] * B[2, :]) ---
        # A index: j=2
        # B index: j=2 (row of B)
        
        # C_0, k1 = A_0,2 * B_2, k1
        U[0*m + 2][col] = 1
        V[2*p + k1][col] = 1
        W[k1*n + 0][col] = 1
        col += 1
        
        # C_0, k2 = A_0,2 * B_2, k2
        U[0*m + 2][col] = 1
        V[2*p + k2][col] = 1
        W[k2*n + 0][col] = 1
        col += 1
        
        # C_1, k1 = A_1,2 * B_2, k1
        U[1*m + 2][col] = 1
        V[2*p + k1][col] = 1
        W[k1*n + 1][col] = 1
        col += 1
        
        # C_1, k2 = A_1,2 * B_2, k2
        U[1*m + 2][col] = 1
        V[2*p + k2][col] = 1
        W[k2*n + 1][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
