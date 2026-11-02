import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 2: A[2x1] * B[1x9] (Rank 18)
    # A indices: i in {0,1}, j=2
    # B indices: j=2, k in {0..8}
    # C indices: i in {0,1}, k in {0..8}
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 1: A[2x2] * B[2x9] (Rank 32)
    # A indices: i in {0,1}, j in {0,1}
    # B indices: j in {0,1}, k in {0..8}
    # C indices: i in {0,1}, k in {0..8}
    
    # 4 blocks of 2x2, 1 block of 2x1
    
    def add_strassen_block(k_start):
        nonlocal col
        # B columns k_start, k_start+1
        k0 = k_start
        k1 = k_start + 1
        
        # M1 = (A00+A11)(B00+B11)
        U[0 * m + 0][col] = 1; U[1 * m + 1][col] = 1
        V[0 * p + k0][col] = 1; V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1; W[k1 * n + 1][col] = 1
        W[k0 * n + 1][col] = 1; W[k1 * n + 0][col] = -1 # Wait, Strassen formulas
        # Let's use the standard Strassen coefficients carefully.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1: A(0,0)+A(1,1) * B(0,0)+B(1,1)
        # U: (0,0)=1, (1,1)=1
        # V: (0,k0)=1, (1,k1)=1
        # W: (k0,0)=1, (k1,1)=1, (k0,1)=1, (k1,0)=1 ??
        # No, W[k*n+i, c] corresponds to C[i,k].
        # M1 contributes to C00, C01, C10, C11?
        # M1 is a scalar. It is added to C00 and C11.
        # So W[k0*n+0] += 1, W[k1*n+1] += 1.
        
        # Let's reset and do it systematically.
        
    # Reset col for Part 1
    col_part1_start = col
    
    # We will define the 7 Strassen multipliers for a generic 2x2 block
    # and apply them 4 times.
    
    def apply_strassen(k0, k1):
        nonlocal col
        # M1 = (A00+A11) * (B00+B11)
        U[0*m+0][col] += 1; U[1*m+1][col] += 1
        V[0*p+k0][col] += 1; V[1*p+k1][col] += 1
        W[k0*n+0][col] += 1; W[k1*n+1][col] += 1
        col += 1
        
        # M2 = (A10+A11) * B00
        U[1*m+0][col] += 1; U[1*m+1][col] += 1
        V[0*p+k0][col] += 1
        W[k0*n+1][col] += 1
        col += 1
        
        # M3 = A00 * (B10-B11)
        U[0*m+0][col] += 1
        V[1*p+k0][col] += 1; V[1*p+k1][col] -= 1
        W[k0*n+0][col] += 1
        col += 1
        
        # M4 = A11 * (B01-B00)
        U[1*m+1][col] += 1
        V[0*p+k1][col] += 1; V[0*p+k0][col] -= 1
        W[k1*n+0][col] += 1; W[k0*n+1][col] += 1
        col += 1
        
        # M5 = (A00+A10) * B11
        U[0*m+0][col] += 1; U[1*m+0][col] += 1
        V[1*p+k1][col] += 1
        W[k1*n+0][col] += 1; W[k1*n+1][col] += 1 # Wait, M5 contributes to C01 and C11?
        # C01 = M3 + M5. C11 = M1 - M2 + M3 + M6.
        # M5 is A00+A10 times B11.
        # B11 is at (1,1) of the block. So V[1*p+k1].
        # Result is added to C01 (i=0, k=k1) and C11 (i=1, k=k1)?
        # Let's check Strassen:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # So M5 contributes to C01 and C00?
        # C00 has -M5. C01 has +M5.
        # So W[k1*n+0] -= 1, W[k1*n+1] += 1?
        # No, M5 is a scalar.
        # M5 = (A00+A10) B11.
        # It appears in C00 with -1 and C01 with +1.
        # So W[k1*n+0] -= 1, W[k1*n+1] += 1.
        
        # I will re-verify the Strassen signs in the code.
        
    # This manual construction is error-prone.
    # I will use a known correct set of coefficients.
    
    # Correct Strassen Coefficients for M[i,j] (i,j in 0,1)
    # M1: A(0,0)+A(1,1) * B(0,0)+B(1,1) -> C(0,0)+1, C(1,1)+1
    # M2: A(1,0)+A(1,1) * B(0,0)       -> C(1,0)+1
    # M3: A(0,0) * B(1,0)-B(1,1)        -> C(0,1)+1, C(0,0)-1? No.
    
    # Let's use the Wikipedia version:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Mapping to 0-indexed:
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # I will implement this carefully.
    
    # Reset U,V,W to 0 for Part 1? No, just continue adding.
    # But I need to be sure Part 2 is correct.
    
    return {"U": U, "V": "V", "W": W}
