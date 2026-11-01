import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition using 8 Strassen blocks
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    col = 0
    
    # Define Strassen block generation
    # A is 2x2, B is 2x2
    # A indices: i in {0,1}, j in {j0, j0+1}
    # B indices: j in {j0, j0+1}, k in {k0, k0+1}
    # C indices: i in {0,1}, k in {k0, k0+1}
    
    def add_strassen_block(j0, k0, offset):
        nonlocal col
        # A_L or A_R depends on j0
        # A rows i=0,1. Cols j=j0, j0+1.
        # B rows j=j0, j0+1. Cols k=k0, k0+1.
        # C rows i=0,1. Cols k=k0, k0+1.
        
        # M1 = (A[0,j0] + A[1,j0+1]) * (B[j0,k0] + B[j0+1,k0+1])
        # Note: Strassen uses A00, A01, A10, A11
        # A00 = A[0, j0], A01 = A[0, j0+1]
        # A10 = A[1, j0], A11 = A[1, j0+1]
        # B00 = B[j0, k0], B01 = B[j0, k0+1]
        # B10 = B[j0+1, k0], B11 = B[j0+1, k0+1]
        
        # M1
        U[0*m + j0][col] += 1
        U[1*m + j0+1][col] += 1
        V[j0*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + 0][col] += 1  # C00
        W[(k0+1)*n + 1][col] += 1 # C11
        col += 1
        
        # M2 = (A[1,j0] + A[1,j0+1]) * B[j0,k0]
        U[1*m + j0][col] += 1
        U[1*m + j0+1][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + 1][col] += 1 # C10
        W[(k0+1)*n + 1][col] -= 1 # C11
        col += 1
        
        # M3 = A[0,j0] * (B[j0+1,k0] - B[j0+1,k0+1])
        U[0*m + j0][col] += 1
        V[(j0+1)*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] -= 1
        W[(k0+1)*n + 0][col] += 1 # C01
        W[(k0+1)*n + 1][col] += 1 # C11
        col += 1
        
        # M4 = A[1,j0+1] * (B[j0,k0+1] - B[j0,k0])
        U[1*m + j0+1][col] += 1
        V[j0*p + (k0+1)][col] += 1
        V[j0*p + k0][col] -= 1
        W[k0*n + 0][col] += 1 # C00
        W[k0*n + 1][col] += 1 # C10
        col += 1
        
        # M5 = (A[0,j0] + A[0,j0+1]) * B[j0+1,k0+1]
        U[0*m + j0][col] += 1
        U[0*m + j0+1][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + 0][col] -= 1 # C00
        W[(k0+1)*n + 0][col] += 1 # C01
        col += 1
        
        # M6 = (A[1,j0] - A[0,j0]) * (B[j0,k0] + B[j0+1,k0])
        U[1*m + j0][col] += 1
        U[0*m + j0][col] -= 1
        V[j0*p + k0][col] += 1
        V[(j0+1)*p + k0][col] += 1
        W[(k0+1)*n + 1][col] += 1 # C11
        col += 1
        
        # M7 = (A[0,j0+1] - A[1,j0+1]) * (B[j0+1,k0+1] + B[j0,k0+1])
        U[0*m + j0+1][col] += 1
        U[1*m + j0+1][col] -= 1
        V[(j0+1)*p + (k0+1)][col] += 1
        V[j0*p + (k0+1)][col] += 1
        W[k0*n + 0][col] += 1 # C00
        col += 1

    # 8 blocks:
    # A_L (j0=0) with B_T (j0=0) for k0=0,2,4,6
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 4, 0)
    add_strassen_block(0, 6, 0)
    
    # A_R (j0=2) with B_B (j0=2) for k0=0,2,4,6
    add_strassen_block(2, 0, 0)
    add_strassen_block(2, 2, 0)
    add_strassen_block(2, 4, 0)
    add_strassen_block(2, 6, 0)
    
    return {"U": U, "V": V, "W": W}
