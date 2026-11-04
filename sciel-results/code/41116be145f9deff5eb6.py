"""Rank-reduced exact decomposition for 3x4 times 4x4 matrix multiplication."""
import time

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Exact rank-44 decomposition for C = A(3x4) * B(4x4).
    
    Uses Strassen's 2x2x2 rank-7 decomposition on four 2x2x2 sub-blocks
    involving A-rows {0,1}, and schoolbook for A-row 2.
    Total: 4*7 + 16 = 44.
    """
    n, m, p = 3, 4, 4
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, i1, j0, j1, k0, k1):
        nonlocal col
        # M1 = (A[i0,j0]+A[i1,j1])(B[j0,k0]+B[j1,k1])
        U[i0*m+j0][col] += 1
        U[i1*m+j1][col] += 1
        V[j0*p+k0][col] += 1
        V[j1*p+k1][col] += 1
        W[k0*n+i0][col] += 1
        W[k1*n+i1][col] += 1
        col += 1
        
        # M2 = (A[i1,j0]+A[i1,j1])B[j0,k0]
        U[i1*m+j0][col] += 1
        U[i1*m+j1][col] += 1
        V[j0*p+k0][col] += 1
        W[k0*n+i1][col] += 1
        W[k1*n+i1][col] += -1
        col += 1
        
        # M3 = A[i0,j0](B[j0,k1]-B[j1,k1])
        U[i0*m+j0][col] += 1
        V[j0*p+k1][col] += 1
        V[j1*p+k1][col] += -1
        W[k1*n+i0][col] += 1
        W[k1*n+i1][col] += 1
        col += 1
        
        # M4 = A[i1,j1](B[j1,k0]-B[j0,k0])
        U[i1*m+j1][col] += 1
        V[j1*p+k0][col] += 1
        V[j0*p+k0][col] += -1
        W[k0*n+i0][col] += 1
        W[k0*n+i1][col] += 1
        col += 1
        
        # M5 = (A[i0,j0]+A[i0,j1])B[j1,k1]
        U[i0*m+j0][col] += 1
        U[i0*m+j1][col] += 1
        V[j1*p+k1][col] += 1
        W[k1*n+i0][col] += 1
        W[k0*n+i0][col] += -1
        col += 1
        
        # M6 = (A[i1,j0]-A[i0,j0])(B[j0,k0]+B[j0,k1])
        U[i1*m+j0][col] += 1
        U[i0*m+j0][col] += -1
        V[j0*p+k0][col] += 1
        V[j0*p+k1][col] += 1
        W[k1*n+i1][col] += 1
        col += 1
        
        # M7 = (A[i0,j1]-A[i1,j1])(B[j1,k0]+B[j1,k1])
        U[i0*m+j1][col] += 1
        U[i1*m+j1][col] += -1
        V[j1*p+k0][col] += 1
        V[j1*p+k1][col] += 1
        W[k0*n+i0][col] += 1
        col += 1
    
    # Four Strassen blocks for A-rows {0,1}
    add_strassen_block(0, 1, 0, 1, 0, 1)
    add_strassen_block(0, 1, 0, 1, 2, 3)
    add_strassen_block(0, 1, 2, 3, 0, 1)
    add_strassen_block(0, 1, 2, 3, 2, 3)
    
    # Schoolbook for A-row 2
    for j in range(m):
        for k in range(p):
            U[2*m+j][col] = 1
            V[j*p+k][col] = 1
            W[k*n+2][col] = 1
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
