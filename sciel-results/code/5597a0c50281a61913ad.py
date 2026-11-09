"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 55
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: j=2 (index 2) part: 2x1x10
    # A[i, 2] * B[2, k] -> C[i, k]
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: j in {0, 1} part: 2x2x10
    # Split k into 5 pairs of size 2
    # Each 2x2x2 block uses Strassen (rank 7)
    for c in range(p // 2):
        k0 = 2 * c
        k1 = 2 * c + 1
        
        # M1 = (A00 + A11)(B00 + B11)
        # U: A00 (i=0,j=0 -> 0), A11 (i=1,j=1 -> 4)
        # V: B00 (j=0,k=k0 -> k0), B11 (j=1,k=k1 -> 10+k1)
        # W: C00 (k=k0,i=0 -> k0*2), C10 (k=k0,i=1 -> k0*2+1), 
        #    C01 (k=k1,i=0 -> k1*2), C11 (k=k1,i=1 -> k1*2+1)
        # C00 += M1, C10 -= M1, C01 += M1, C11 -= M1
        U[0][col] += 1
        U[4][col] += 1
        V[k0][col] += 1
        V[10 + k1][col] += 1
        W[k0 * n + 0][col] += 1
        W[k0 * n + 1][col] -= 1
        W[k1 * n + 0][col] += 1
        W[k1 * n + 1][col] -= 1
        col += 1
        
        # M2 = (A10 + A11) B00
        # U: A10 (i=1,j=0 -> 3), A11 (i=1,j=1 -> 4)
        # V: B00 (j=0,k=k0 -> k0)
        # W: C10 += M2, C01 -= M2, C11 += M2
        U[3][col] += 1
        U[4][col] += 1
        V[k0][col] += 1
        W[k0 * n + 1][col] += 1
        W[k1 * n + 0][col] -= 1
        W[k1 * n + 1][col] += 1
        col += 1
        
        # M3 = A00 (B01 - B11)
        # U: A00 (i=0,j=0 -> 0)
        # V: B01 (j=0,k=k1 -> k1), B11 (j=1,k=k1 -> 10+k1)
        # W: C00 -= M3, C01 += M3, C11 += M3
        U[0][col] += 1
        V[k1][col] += 1
        V[10 + k1][col] -= 1
        W[k0 * n + 0][col] -= 1
        W[k1 * n + 0][col] += 1
        W[k1 * n + 1][col] += 1
        col += 1
        
        # M4 = A11 (B10 - B00)
        # U: A11 (i=1,j=1 -> 4)
        # V: B10 (j=1,k=k0 -> 10+k0), B00 (j=0,k=k0 -> k0)
        # W: C00 += M4, C10 += M4, C11 -= M4
        U[4][col] += 1
        V[10 + k0][col] += 1
        V[k0][col] -= 1
        W[k0 * n + 0][col] += 1
        W[k0 * n + 1][col] += 1
        W[k1 * n + 1][col] -= 1
        col += 1
        
        # M5 = (A00 + A01) B11
        # U: A00 (i=0,j=0 -> 0), A01 (i=0,j=1 -> 1)
        # V: B11 (j=1,k=k1 -> 10+k1)
        # W: C00 -= M5, C10 += M5, C11 -= M5
        U[0][col] += 1
        U[1][col] += 1
        V[10 + k1][col] += 1
        W[k0 * n + 0][col] -= 1
        W[k0 * n + 1][col] += 1
        W[k1 * n + 1][col] -= 1
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        # U: A10 (i=1,j=0 -> 3), A00 (i=0,j=0 -> 0)
        # V: B00 (j=0,k=k0 -> k0), B01 (j=0,k=k1 -> k1)
        # W: C10 -= M6, C01 += M6
        U[3][col] += 1
        U[0][col] -= 1
        V[k0][col] += 1
        V[k1][col] += 1
        W[k0 * n + 1][col] -= 1
        W[k1 * n + 0][col] += 1
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        # U: A01 (i=0,j=1 -> 1), A11 (i=1,j=1 -> 4)
        # V: B10 (j=1,k=k0 -> 10+k0), B11 (j=1,k=k1 -> 10+k1)
        # W: C00 += M7, C11 += M7
        U[1][col] += 1
        U[4][col] -= 1
        V[10 + k0][col] += 1
        V[10 + k1][col] += 1
        W[k0 * n + 0][col] += 1
        W[k1 * n + 1][col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
