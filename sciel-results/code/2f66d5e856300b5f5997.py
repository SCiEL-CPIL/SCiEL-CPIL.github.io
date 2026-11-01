"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]  # 6 x 28
    V = [[0.0] * rank for _ in range(m * p)]  # 15 x 28
    W = [[0.0] * rank for _ in range(p * n)]  # 10 x 28
    
    col = 0
    
    def add_strassen_222(i0, j0, k0):
        nonlocal col
        # P1 = (A[i0,j0]+A[i0+1,j0+1])(B[j0,k0]+B[j0+1,k0+1])
        U[i0*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # P2 = (A[i0+1,j0]+A[i0+1,j0+1]) * B[j0,k0]
        U[(i0+1)*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        col += 1
        
        # P3 = A[i0,j0] * (B[j0,k0+1] - B[j0+1,k0+1])
        U[i0*m + j0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += -1
        W[(k0+1)*n + i0][col] += 1
        col += 1
        
        # P4 = A[i0+1,j0+1] * (B[j0+1,k0] - B[j0,k0])
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+1)*p + k0][col] += 1
        V[j0*p + k0][col] += -1
        W[k0*n + (i0+1)][col] += 1
        col += 1
        
        # P5 = (A[i0,j0]+A[i0,j0+1]) * B[j0+1,k0+1]
        U[i0*m + j0][col] += 1
        U[i0*m + (j0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[(k0+1)*n + i0][col] += 1
        col += 1
        
        # P6 = (A[i0+1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k0+1])
        U[(i0+1)*m + j0][col] += 1
        U[i0*m + j0][col] += -1
        V[j0*p + k0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        W[k0*n + (i0+1)][col] += 1
        W[(k0+1)*n + i0][col] += -1
        col += 1
        
        # P7 = (A[i0,j0+1]-A[i0+1,j0+1]) * (B[j0+1,k0]+B[j0+1,k0+1])
        U[i0*m + (j0+1)][col] += 1
        U[(i0+1)*m + (j0+1)][col] += -1
        V[(j0+1)*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + (i0+1)][col] += -1
        W[(k0+1)*n + i0][col] += 1
        col += 1
    
    def add_schoolbook_221(i0, j0, k0):
        nonlocal col
        for i in range(2):
            for j in range(2):
                U[(i0+i)*m + (j0+j)][col] += 1
                V[(j0+j)*p + k0][col] += 1
                W[k0*n + (i0+i)][col] += 1
                col += 1
    
    def add_schoolbook_215(i0, j0, k0):
        nonlocal col
        for i in range(2):
            for k in range(5):
                U[(i0+i)*m + j0][col] += 1
                V[j0*p + (k0+k)][col] += 1
                W[(k0+k)*n + (i0+i)][col] += 1
                col += 1
    
    # Part 1a: A[0:2, 0:2] * B[0:2, 0:2] - Strassen
    add_strassen_222(0, 0, 0)
    
    # Part 1b: A[0:2, 0:2] * B[0:2, 2:4] - Strassen
    add_strassen_222(0, 0, 2)
    
    # Part 1c: A[0:2, 0:2] * B[0:2, 4:5] - Schoolbook 2x2x1
    add_schoolbook_221(0, 0, 4)
    
    # Part 2: A[0:2, 2:3] * B[2:3, 0:5] - Schoolbook 2x1x5
    add_schoolbook_215(0, 2, 0)
    
    assert col == rank, f"col={col}, rank={rank}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
