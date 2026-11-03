"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Start with schoolbook rank, then replace 8 terms with 7 Strassen terms
    # Schoolbook rank is n*m*p = 60
    # We replace the 2x2x2 sub-block at i in {0,1}, j in {0,1}, k in {0,1}
    # This removes 8 terms and adds 7, resulting in rank 59.
    
    # Indices for the 2x2x2 sub-block
    # i in 0..1, j in 0..1, k in 0..1
    
    # We will construct U, V, W for rank 59
    rank = 59
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 1. Add the 52 schoolbook terms that are NOT in the 2x2x2 sub-block
    # Sub-block condition: i in {0,1}, j in {0,1}, k in {0,1}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if in the 2x2x2 corner
                if i < 2 and j < 2 and k < 2:
                    continue
                
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # 2. Add the 7 Strassen terms for the 2x2x2 sub-block
    # The Strassen terms operate on A[0:2, 0:2], B[0:2, 0:2]
    # and produce C[0:2, 0:2]
    
    # Term 1: M1 = (A00 + A11)(B00 + B11)
    # U: A00, A11 -> 1
    # V: B00, B11 -> 1
    # W: C00, C11 -> 1
    U[0 * m + 0][col] = 1  # A00
    U[1 * m + 1][col] = 1  # A11
    V[0 * p + 0][col] = 1  # B00
    V[1 * p + 1][col] = 1  # B11
    W[0 * n + 0][col] = 1  # C00
    W[1 * n + 1][col] = 1  # C11
    col += 1
    
    # Term 2: M2 = (A10 + A11)B00
    # U: A10, A11 -> 1
    # V: B00 -> 1
    # W: C10 -> 1, C11 -> -1
    U[1 * m + 0][col] = 1  # A10
    U[1 * m + 1][col] = 1  # A11
    V[0 * p + 0][col] = 1  # B00
    W[0 * n + 1][col] = 1  # C10 (index k*n+i, k=0, i=1)
    W[1 * n + 1][col] = -1 # C11 (index k*n+i, k=1, i=1)
    col += 1
    
    # Term 3: M3 = A00(B10 - B11)
    # U: A00 -> 1
    # V: B10 -> 1, B11 -> -1
    # W: C01 -> 1, C11 -> 1
    U[0 * m + 0][col] = 1  # A00
    V[1 * p + 0][col] = 1  # B10
    V[1 * p + 1][col] = -1 # B11
    W[1 * n + 0][col] = 1  # C01 (index k*n+i, k=1, i=0)
    W[1 * n + 1][col] = 1  # C11
    col += 1
    
    # Term 4: M4 = A11(B01 - B00)
    # U: A11 -> 1
    # V: B01 -> 1, B00 -> -1
    # W: C00 -> 1, C10 -> 1
    U[1 * m + 1][col] = 1  # A11
    V[0 * p + 1][col] = 1  # B01
    V[0 * p + 0][col] = -1 # B00
    W[0 * n + 0][col] = 1  # C00
    W[0 * n + 1][col] = 1  # C10
    col += 1
    
    # Term 5: M5 = (A00 + A01)B11
    # U: A00, A01 -> 1
    # V: B11 -> 1
    # W: C00 -> -1, C01 -> 1
    U[0 * m + 0][col] = 1  # A00
    U[0 * m + 1][col] = 1  # A01
    V[1 * p + 1][col] = 1  # B11
    W[0 * n + 0][col] = -1 # C00
    W[1 * n + 0][col] = 1  # C01
    col += 1
    
    # Term 6: M6 = (A10 - A00)(B00 + B01)
    # U: A10 -> 1, A00 -> -1
    # V: B00, B01 -> 1
    # W: C11 -> 1
    U[1 * m + 0][col] = 1  # A10
    U[0 * m + 0][col] = -1 # A00
    V[0 * p + 0][col] = 1  # B00
    V[0 * p + 1][col] = 1  # B01
    W[1 * n + 1][col] = 1  # C11
    col += 1
    
    # Term 7: M7 = (A01 - A11)(B10 + B11)
    # U: A01 -> 1, A11 -> -1
    # V: B10, B11 -> 1
    # W: C00 -> 1, C11 -> 1
    U[0 * m + 1][col] = 1  # A01
    U[1 * m + 1][col] = -1 # A11
    V[1 * p + 0][col] = 1  # B10
    V[1 * p + 1][col] = 1  # B11
    W[0 * n + 0][col] = 1  # C00
    W[1 * n + 1][col] = 1  # C11
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
