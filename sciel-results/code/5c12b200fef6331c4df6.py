"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=5, p=5
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # 10 x 49
    V = [[0] * rank for _ in range(m * p)]  # 25 x 49
    W = [[0] * rank for _ in range(p * n)]  # 10 x 49
    
    col = 0
    
    # U indexing: U[i*m + j] for A[i,j]
    # V indexing: V[j*p + k] for B[j,k]
    # W indexing: W[k*n + i] for C[i,k]
    
    # A1 = A[0:2, 0:2], A2 = A[0:2, 2:5]
    # B1 = B[0:2, :5], B2 = B[2:5, :5]
    # B11 = B[0:2, 0:2], B12 = B[0:2, 2:5]
    # B21 = B[2:5, 0:2], B22 = B[2:5, 2:5]
    
    # Strassen for A1*B11 (2x2 * 2x2)
    # A11=A[0,0]=U[0], A12=A[0,1]=U[1], A21=A[1,0]=U[2], A22=A[1,1]=U[3]
    # B11=B[0,0]=V[0], B12=B[0,1]=V[1], B21=B[1,0]=V[5], B22=B[1,1]=V[6]
    # C11=C[0,0]=W[0], C12=C[0,1]=W[2], C21=C[1,0]=W[1], C22=C[1,1]=W[3]
    
    # M1 = (A11+A22)(B11+B22)
    U[0][col] = 1; U[3][col] = 1
    V[0][col] = 1; V[6][col] = 1
    W[0][col] = 1; W[3][col] = 1
    col += 1
    
    # M2 = (A21+A22)B11
    U[2][col] = 1; U[3][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[3][col] = -1
    col += 1
    
    # M3 = A11(B12-B22)
    U[0][col] = 1
    V[1][col] = 1; V[6][col] = -1
    W[2][col] = 1; W[3][col] = 1
    col += 1
    
    # M4 = A22(B21-B11)
    U[3][col] = 1
    V[5][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # M5 = (A11+A12)B22
    U[0][col] = 1; U[1][col] = 1
    V[6][col] = 1
    W[0][col] = -1; W[2][col] = 1
    col += 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[3][col] = 1
    col += 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1][col] = 1; U[3][col] = -1
    V[5][col] = 1; V[6][col] = 1
    W[0][col] = 1
    col += 1
    
    # A2*B21 (2x3 * 3x2): 12 terms
    for i in range(2):
        for j in range(3):
            for k in range(2):
                U[i * m + (j + 2)][col] = 1
                V[(j + 2) * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # A1*B12 (2x2 * 2x3): 12 terms
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + (k + 2)][col] = 1
                W[(k + 2) * n + i][col] = 1
                col += 1
    
    # A2*B22 (2x3 * 3x3): 18 terms
    for i in range(2):
        for j in range(3):
            for k in range(3):
                U[i * m + (j + 2)][col] = 1
                V[(j + 2) * p + (k + 2)][col] = 1
                W[(k + 2) * n + i][col] = 1
                col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
