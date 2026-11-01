import time

DIMENSIONS = [2, 2, 9]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=9
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen parameters for 2x2 blocks
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # U indices: i*m + j -> i*2 + j
    # V indices: j*p + k -> j*9 + k
    # W indices: k*n + i -> k*2 + i
    
    strassen_blocks = 4
    
    for t in range(strassen_blocks):
        # k offset for this block
        k_off = t * 2
        
        # M1
        U[0][col] = 1 # A00
        U[3][col] = 1 # A11
        V[0*9 + k_off][col] = 1 # B0, k_off
        V[1*9 + k_off+1][col] = 1 # B1, k_off+1
        W[(k_off)*2 + 0][col] = 1 # C00
        W[(k_off)*2 + 1][col] = -1 # C01
        W[(k_off+1)*2 + 0][col] = 1 # C10
        W[(k_off+1)*2 + 1][col] = 1 # C11
        col += 1
        
        # M2
        U[2][col] = 1 # A10
        U[3][col] = 1 # A11
        V[0*9 + k_off][col] = 1 # B0, k_off
        W[(k_off+1)*2 + 0][col] = 1 # C10
        W[(k_off+1)*2 + 1][col] = -1 # C11
        col += 1
        
        # M3
        U[0][col] = 1 # A00
        V[0*9 + k_off+1][col] = 1 # B0, k_off+1
        V[1*9 + k_off+1][col] = -1 # B1, k_off+1
        W[(k_off)*2 + 1][col] = 1 # C01
        W[(k_off+1)*2 + 1][col] = 1 # C11
        col += 1
        
        # M4
        U[3][col] = 1 # A11
        V[1*9 + k_off][col] = 1 # B1, k_off
        V[0*9 + k_off][col] = -1 # B0, k_off
        W[(k_off)*2 + 0][col] = 1 # C00
        W[(k_off+1)*2 + 0][col] = 1 # C10
        col += 1
        
        # M5
        U[0][col] = 1 # A00
        U[1][col] = 1 # A01
        V[1*9 + k_off+1][col] = 1 # B1, k_off+1
        W[(k_off)*2 + 0][col] = -1 # C00
        W[(k_off)*2 + 1][col] = 1 # C01
        col += 1
        
        # M6
        U[2][col] = 1 # A10
        U[0][col] = -1 # A00
        V[0*9 + k_off][col] = 1 # B0, k_off
        V[0*9 + k_off+1][col] = 1 # B0, k_off+1
        W[(k_off+1)*2 + 1][col] = 1 # C11
        col += 1
        
        # M7
        U[1][col] = 1 # A01
        U[3][col] = -1 # A11
        V[1*9 + k_off][col] = 1 # B1, k_off
        V[1*9 + k_off+1][col] = 1 # B1, k_off+1
        W[(k_off)*2 + 0][col] = 1 # C00
        W[(k_off+1)*2 + 1][col] = 1 # C11
        col += 1
        
    # Last block t=4, k=8, 2x1
    # C80 = A00 B08 + A01 B18
    # C81 = A10 B08 + A11 B18
    
    # M8: A00 B08
    U[0][col] = 1
    V[0*9 + 8][col] = 1
    W[8*2 + 0][col] = 1
    col += 1
    
    # M9: A01 B18
    U[1][col] = 1
    V[1*9 + 8][col] = 1
    W[8*2 + 0][col] = 1
    col += 1
    
    # M10: A10 B08
    U[2][col] = 1
    V[0*9 + 8][col] = 1
    W[8*2 + 1][col] = 1
    col += 1
    
    # M11: A11 B18
    U[3][col] = 1
    V[1*9 + 8][col] = 1
    W[8*2 + 1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
