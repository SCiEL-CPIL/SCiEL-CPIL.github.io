import time

def construct():
    start_time = time.time()
    
    # Rank 26 decomposition for 3x3 matrix multiplication
    # Using Strassen for the 2x2 core and explicit terms for the border
    # Indices:
    # U: A[i][j] at i*3 + j
    # V: B[j][k] at j*3 + k
    # W: C[k][i] at k*3 + i (Trace convention)
    
    rank = 26
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(9)]
    W = [[0] * rank for _ in range(9)]
    
    # Helper to add a term
    def add_term(u, v, w, coeff=1):
        nonlocal col
        U[u][col] += coeff
        V[v][col] = 1
        W[w][col] = 1
        col += 1
    
    col = 0
    
    # Core 2x2 Strassen (Rank 7)
    # A00=0, A01=1, A10=3, A11=4
    # B00=0, B01=1, B10=3, B11=4
    # C00=W0, C01=W3, C10=W1, C11=W4
    
    # M1 = (A00+A11)(B00+B11)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    # M2 = (A10+A11)B00
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1
    
    # M3 = A00(B01-B11)
    U[0][col] = 1
    V[1][col] = 1; V[4][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    
    # M4 = A11(B10-B00)
    U[4][col] = 1
    V[3][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # M5 = (A00+A01)B11
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1
    
    # M6 = (A10-A00)(B00+B01)
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][col] = 1; U[4][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[0][col] = 1
    col += 1
    
    # Border Terms (Rank 19)
    # Indices for Border:
    # A: 02=2, 12=5, 20=6, 21=7, 22=8
    # B: 02=2, 12=5, 20=6, 21=7, 22=8
    
    # Block 2: A[0:2, 2] * B[2, 0:2]
    add_term(2, 6, 0, 1) # A02 B20 -> C00
    add_term(2, 7, 3, 1) # A02 B21 -> C01
    add_term(5, 6, 1, 1) # A12 B20 -> C10
    add_term(5, 7, 4, 1) # A12 B21 -> C11
    
    # Block 3: A[0:2, 0:2] * B[0:2, 2]
    add_term(0, 2, 6, 1) # A00 B02 -> C02
    add_term(1, 5, 6, 1) # A01 B12 -> C02
    add_term(3, 2, 7, 1) # A10 B02 -> C12
    add_term(4, 5, 7, 1) # A11 B12 -> C12
    
    # Block 4: A[0:2, 2] * B[2, 2]
    add_term(2, 8, 6, 1) # A02 B22 -> C02
    add_term(5, 8, 7, 1) # A12 B22 -> C12
    
    # Block 5: A[2, 0:2] * B[0:2, 0:2]
    add_term(6, 0, 2, 1) # A20 B00 -> C20
    add_term(7, 3, 2, 1) # A21 B10 -> C20
    add_term(6, 1, 5, 1) # A20 B01 -> C21
    add_term(7, 4, 5, 1) # A21 B11 -> C21
    
    # Block 6: A[2, 2] * B[2, 0:2]
    add_term(8, 6, 2, 1) # A22 B20 -> C20
    add_term(8, 7, 5, 1) # A22 B21 -> C21
    
    # Block 7: A[2, 0:2] * B[0:2, 2]
    add_term(6, 2, 8, 1) # A20 B02 -> C22
    add_term(7, 5, 8, 1) # A21 B12 -> C22
    
    # Block 8: A[2, 2] * B[2, 2]
    add_term(8, 8, 8, 1) # A22 B22 -> C22
    
    return {"U": U, "V": V, "W": W}
