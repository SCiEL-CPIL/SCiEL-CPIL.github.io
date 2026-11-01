"""Program to find a rank 29 decomposition for 2x3x5 matrix multiplication."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Dimensions: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # n=2, m=3, p=5
    # U size: 6
    # V size: 15
    # W size: 10
    
    rank = 29
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 1. Strassen decomposition for subblock A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A indices in U: i*m + j. i in {0,1}, j in {0,1} -> 0, 1, 3, 4
    # B indices in V: j*p + k. j in {0,1}, k in {0,1} -> 0, 1, 5, 6
    # C indices in W: k*n + i. k in {0,1}, i in {0,1} -> 0, 1, 2, 3
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[6][col] = 1
    W[0][col] = 1; W[3][col] = 1
    col += 1
    
    # M2 = (A00 + A11)B10
    U[0][col] = 1; U[4][col] = 1
    V[5][col] = 1
    W[1][col] = 1  # C10
    col += 1
    
    # M3 = A00(B01 - B11)
    U[0][col] = 1
    V[1][col] = 1; V[6][col] = -1
    W[2][col] = 1  # C01
    col += 1
    
    # M4 = A11(B10 - B00)
    U[4][col] = 1
    V[5][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1  # C00, C10
    col += 1
    
    # M5 = (A01 + A11)B00
    U[1][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[0][col] = -1; W[2][col] = 1  # -C00, +C01
    col += 1
    
    # M6 = (A00 - A10)(B10 + B11)
    U[0][col] = 1; U[3][col] = -1
    V[5][col] = 1; V[6][col] = 1
    W[3][col] = 1  # C11
    col += 1
    
    # M7 = (A10 - A00)(B00 + B01)
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[0][col] = 1  # C00
    col += 1
    
    # 2. Terms for A[i,2] * B[2,k] for k in {0,1}
    # A index: i*3 + 2 -> 2, 5
    # B index: 2*5 + k -> 10, 11
    # C index: k*2 + i -> 0, 2 (k=0); 1, 3 (k=1)
    
    # i=0, k=0: A02 * B20 -> C00
    U[2][col] = 1
    V[10][col] = 1
    W[0][col] = 1
    col += 1
    
    # i=0, k=1: A02 * B21 -> C01
    U[2][col] = 1
    V[11][col] = 1
    W[2][col] = 1
    col += 1
    
    # i=1, k=0: A12 * B20 -> C10
    U[5][col] = 1
    V[10][col] = 1
    W[1][col] = 1
    col += 1
    
    # i=1, k=1: A12 * B21 -> C11
    U[5][col] = 1
    V[11][col] = 1
    W[3][col] = 1
    col += 1
    
    # 3. Schoolbook for k in {2,3,4} (cols 3,4,5)
    # C[i, k] = sum_j A[i, j] B[j, k]
    for i in range(n):
        for j in range(m):
            for k in range(2, p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
