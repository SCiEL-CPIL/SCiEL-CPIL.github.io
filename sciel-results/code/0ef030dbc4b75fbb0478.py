import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=6
    
    # We will build U, V, W matrices
    # U shape: (n*m, rank) = (6, 33)
    # V shape: (m*p, rank) = (18, 33)
    # W shape: (p*n, rank) = (12, 33)
    
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A[2x2] * B[2x6]
    # A indices: i in 0..1, j in 0..1
    # B indices: j in 0..1, k in 0..5
    # C indices: i in 0..1, k in 0..5
    
    # We use 3 Strassen blocks for the 6 columns
    # Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each block of 2 columns (k, k+1), we apply this.
    
    for block in range(3):
        k0 = block * 2
        k1 = block * 2 + 1
        
        # M1
        # U: A00, A11
        # V: B00, B11 (for k0, k1)
        # W: C00, C11 (for k0, k1)
        
        # Term 1: A00 * (B00+B11)
        U[0*3+0][col] = 1 # A00
        V[0*6+k0][col] = 1 # B00 k0
        V[1*6+k0][col] = 1 # B11 k0
        W[k0*2+0][col] = 1 # C00 k0
        W[k1*2+1][col] = 1 # C11 k1
        col += 1
        
        # Term 2: A11 * (B00+B11)
        U[1*3+1][col] = 1 # A11
        V[0*6+k0][col] = 1 # B00 k0
        V[1*6+k0][col] = 1 # B11 k0
        W[k0*2+0][col] = 1 # C00 k0
        W[k1*2+1][col] = 1 # C11 k1
        col += 1
        
        # Wait, Strassen M1 is (A00+A11)(B00+B11).
        # This is 4 terms if expanded? No, Strassen uses 7 multiplications.
        # The "multiplications" are the rank-1 terms.
        # So M1 is one term: U=(A00+A11), V=(B00+B11), W=(C00, C11).
        # But U and V must be linear combinations.
        # In our representation, U, V, W are matrices.
        # We can set U to have 1 at A00 and A11.
        # V to have 1 at B00 and B11.
        # W to have 1 at C00 and C11.
        # This represents the product (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
        # This is correct for M1.
        
        # Let's restart the Strassen construction properly.
        # I will revert col to 0 and write a helper.
        
    col = 0
    
    def add_strassen_block(k0, k1):
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        # A00 is U[0, col], A11 is U[4, col] (index 1*3+1)
        # B00 is V[0, col] (index 0*6+k0), B11 is V[6+k0, col] (index 1*6+k0)
        # C00 is W[k0, col] (index k0*2+0), C11 is W[6+k1, col] (index k1*2+1)
        
        # M1
        U[0][col] = 1
        U[4][col] = 1
        V[0*6+k0][col] = 1
        V[1*6+k0][col] = 1
        W[k0*2+0][col] = 1
        W[k1*2+1][col] = 1
        col += 1
        
        # M2 = (A10+A11)B00
        # A10 is U[3, col] (1*3+0), A11 is U[4, col]
        # B00 is V[0*6+k0, col]
        # C10 is W[k0*2+1, col]
        
        U[3][col] = 1
        U[4][col] = 1
        V[0*6+k0][col] = 1
        W[k0*2+1][col] = 1
        col += 1
        
        # M3 = A00(B10-B11)
        # A00 is U[0, col]
        # B10 is V[1*6+k0, col], B11 is V[1*6+k1, col]
        # C01 is W[k1*2+0, col]
        
        U[0][col] = 1
        V[1*6+k0][col] = 1
        V[1*6+k1][col] = -1
        W[k1*2+0][col] = 1
        col += 1
        
        # M4 = A11(B01-B00)
        # A11 is U[4, col]
        # B01 is V[0*6+k1, col], B00 is V[0*6+k0, col]
        # C00 is W[k0*2+0, col], C10 is W[k0*2+1, col]
        
        U[4][col] = 1
        V[0*6+k1][col] = 1
        V[0*6+k0][col] = -1
        W[k0*2+0][col] = 1
        W[k0*2+1][col] = 1
        col += 1
        
        # M5 = (A00+A01)B11
        # A00 is U[0, col], A01 is U[1, col]
        # B11 is V[1*6+k1, col]
        # C00 is W[k0*2+0, col], C01 is W[k1*2+0, col]
        
        U[0][col] = 1
        U[1][col] = 1
        V[1*6+k1][col] = 1
        W[k0*2+0][col] = -1 # -M5 in C00
        W[k1*2+0][col] = 1 # +M5 in C01
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        # A10 is U[3, col], A00 is U[0, col]
        # B00 is V[0*6+k0, col], B01 is V[0*6+k1, col]
        # C11 is W[k1*2+1, col]
        
        U[3][col] = 1
        U[0][col] = -1
        V[0*6+k0][col] = 1
        V[0*6+k1][col] = 1
        W[k1*2+1][col] = 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        # A01 is U[1, col], A11 is U[4, col]
        # B10 is V[1*6+k0, col], B11 is V[1*6+k1, col]
        # C00 is W[k0*2+0, col]
        
        U[1][col] = 1
        U[4][col] = -1
        V[1*6+k0][col] = 1
        V[1*6+k1][col] = 1
        W[k0*2+0][col] = 1
        col += 1

    add_strassen_block(0, 1)
    add_strassen_block(2, 3)
    add_strassen_block(4, 5)
    
    # Part 2: a * b
    # a is A[:, 2] (indices 2, 5)
    # b is B[2, :] (indices 12, 13, 14, 15, 16, 17)
    # C is A[i,2] B[2,k] at C[i,k]
    
    for i in range(2):
        for k in range(6):
            U[i*3+2][col] = 1
            V[2*6+k][col] = 1
            W[k*2+i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
