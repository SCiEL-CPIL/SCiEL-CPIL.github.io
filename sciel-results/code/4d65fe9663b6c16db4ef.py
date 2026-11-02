"""
Rewritten program for Matrix Multiplication 2x2x8.
Uses Strassen's algorithm for each 2x2 block of B.
Total rank: 4 blocks * 7 multiplications/block = 28.
"""

def construct():
    n, m, p = 2, 2, 8
    rank = 28
    
    # Initialize U, V, W
    # U: (n*m, rank) = (4, 28)
    # V: (m*p, rank) = (16, 28)
    # W: (p*n, rank) = (16, 28)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for a single 2x2 block
    # A is 2x2, B_block is 2x2, C_block is 2x2
    # Indices for A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for B_block: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    #   In global V (m*p):
    #   B[0, k] -> index 0*2 + k = k
    #   B[1, k] -> index 1*2 + k = 2 + k
    #   For block q (cols 2q, 2q+1):
    #   B[0, 2q] -> 2q
    #   B[0, 2q+1] -> 2q+1
    #   B[1, 2q] -> 2 + 2q
    #   B[1, 2q+1] -> 2 + 2q + 1
    # Indices for C_block: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    #   In global W (p*n): W[k*n + i]
    #   C[0, 2q] -> 2q*2 + 0 = 4q
    #   C[0, 2q+1] -> (2q+1)*2 + 0 = 4q + 2
    #   C[1, 2q] -> 2q*2 + 1 = 4q + 1
    #   C[1, 2q+1] -> (2q+1)*2 + 1 = 4q + 3
    
    for q in range(4):
        # Global column indices for this block
        c0 = 2 * q
        c1 = 2 * q + 1
        
        # Local V indices
        v00 = c0       # B[0, c0]
        v01 = c1       # B[0, c1]
        v10 = 2 + c0   # B[1, c0]
        v11 = 2 + c1   # B[1, c1]
        
        # Local W indices
        w00 = 2 * c0       # C[0, c0] -> 2*c0 + 0
        w01 = 2 * c1       # C[0, c1] -> 2*c1 + 0
        w10 = 2 * c0 + 1   # C[1, c0] -> 2*c0 + 1
        w11 = 2 * c1 + 1   # C[1, c1] -> 2*c1 + 1
        
        # Strassen Terms
        # r is global rank index
        r = 7 * q
        
        # M1 = (A00 + A11) * (B00 + B11)
        # U: A00, A11
        U[0][r] += 1
        U[3][r] += 1
        # V: B00, B11
        V[v00][r] += 1
        V[v11][r] += 1
        # W: C00, C11
        W[w00][r] += 1
        W[w11][r] += 1
        
        r += 1
        # M2 = (A10 + A11) * B00
        U[2][r] += 1
        U[3][r] += 1
        V[v00][r] += 1
        W[w10][r] += 1
        W[w11][r] -= 1
        
        r += 1
        # M3 = A00 * (B01 - B11)
        U[0][r] += 1
        V[v01][r] += 1
        V[v11][r] -= 1
        W[w01][r] += 1
        W[w11][r] += 1
        
        r += 1
        # M4 = A11 * (B10 - B00)
        U[3][r] += 1
        V[v10][r] += 1
        V[v00][r] -= 1
        W[w00][r] += 1
        W[w10][r] += 1
        
        r += 1
        # M5 = (A00 + A01) * B11
        U[0][r] += 1
        U[1][r] += 1
        V[v11][r] += 1
        W[w00][r] -= 1
        W[w01][r] += 1
        
        r += 1
        # M6 = (A10 - A00) * (B00 + B01)
        U[2][r] += 1
        U[0][r] -= 1
        V[v00][r] += 1
        V[v01][r] += 1
        W[w11][r] += 1
        
        r += 1
        # M7 = (A01 - A11) * (B10 + B11)
        U[1][r] += 1
        U[3][r] -= 1
        V[v10][r] += 1
        V[v11][r] += 1
        W[w00][r] += 1

    return {"U": U, "V": V, "W": W}
