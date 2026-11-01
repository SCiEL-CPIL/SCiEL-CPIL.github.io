import time

DIMENSIONS = [2, 2, 11]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=11
    # Rank 39
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 5 Strassen blocks
    for block in range(5):
        # Columns 2*block and 2*block+1
        k1 = 2 * block
        k2 = 2 * block + 1
        
        # M1 = (A00 + A11) * (B0,k1 + B1,k1) -- Wait, Strassen uses B00, B11 etc relative to the 2x2 block.
        # The B matrix for this block is:
        # B0,k1  B0,k2
        # B1,k1  B1,k2
        # So B00 -> B0,k1, B01 -> B0,k2, B10 -> B1,k1, B11 -> B1,k2
        
        # U indices: 0=A00, 1=A01, 2=A10, 3=A11
        # V indices: j*p + k. j=0 -> k, j=1 -> p+k.
        
        # M1
        U[0][col] += 1
        U[3][col] += 1
        V[0 * p + k1][col] += 1
        V[1 * p + k1][col] += 1
        # W: C00 (k1, i=0), C01 (k2, i=0), C10 (k1, i=1), C11 (k2, i=1)
        # C00 = M1 + M4 - M5 + M7
        W[k1 * n + 0][col] += 1
        # C01 = M3 + M5 (M1 not here)
        # C10 = M2 + M4 (M1 not here)
        # C11 = M1 - M2 + M3 + M6
        W[k2 * n + 1][col] += 1
        col += 1
        
        # M2 = (A10 + A11) * B0,k1
        U[2][col] += 1
        U[3][col] += 1
        V[0 * p + k1][col] += 1
        # C10 += M2
        W[k1 * n + 1][col] += 1
        # C11 -= M2
        W[k2 * n + 1][col] -= 1
        col += 1
        
        # M3 = A00 * (B1,k2 - B1,k1)
        U[0][col] += 1
        V[1 * p + k2][col] += 1
        V[1 * p + k1][col] -= 1
        # C01 += M3
        W[k2 * n + 0][col] += 1
        # C11 += M3
        W[k2 * n + 1][col] += 1
        col += 1
        
        # M4 = A11 * (B0,k2 - B0,k1)
        U[3][col] += 1
        V[0 * p + k2][col] += 1
        V[0 * p + k1][col] -= 1
        # C00 += M4
        W[k1 * n + 0][col] += 1
        # C10 += M4
        W[k1 * n + 1][col] += 1
        col += 1
        
        # M5 = (A00 + A01) * B1,k2
        U[0][col] += 1
        U[1][col] += 1
        V[1 * p + k2][col] += 1
        # C00 -= M5
        W[k1 * n + 0][col] -= 1
        # C01 += M5
        W[k2 * n + 0][col] += 1
        col += 1
        
        # M6 = (A10 - A00) * (B0,k1 + B1,k1)
        U[2][col] += 1
        U[0][col] -= 1
        V[0 * p + k1][col] += 1
        V[1 * p + k1][col] += 1
        # C11 += M6
        W[k2 * n + 1][col] += 1
        col += 1
        
        # M7 = (A01 - A11) * (B1,k2 + B0,k2)
        U[1][col] += 1
        U[3][col] -= 1
        V[1 * p + k2][col] += 1
        V[0 * p + k2][col] += 1
        # C00 += M7
        W[k1 * n + 0][col] += 1
        # C11 += M7
        W[k2 * n + 1][col] += 1
        col += 1

    # 1 Schoolbook block for column 10
    k = 10
    # M8 = A00 * B0,k
    U[0][col] += 1
    V[0 * p + k][col] += 1
    W[k * n + 0][col] += 1
    col += 1
    
    # M9 = A01 * B1,k
    U[1][col] += 1
    V[1 * p + k][col] += 1
    W[k * n + 0][col] += 1
    col += 1
    
    # M10 = A10 * B0,k
    U[2][col] += 1
    V[0 * p + k][col] += 1
    W[k * n + 1][col] += 1
    col += 1
    
    # M11 = A11 * B1,k
    U[3][col] += 1
    V[1 * p + k][col] += 1
    W[k * n + 1][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
