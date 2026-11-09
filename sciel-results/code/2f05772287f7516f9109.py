import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We construct a rank-85 decomposition by splitting the 4x5x5 tensor into:
    # 1. 4x4x4 sub-tensor (using rank-49 Strassen decomposition)
    # 2. 4x4x1 sub-tensor (naive, rank 16)
    # 3. 4x1x4 sub-tensor (naive, rank 16)
    # 4. 4x1x1 sub-tensor (naive, rank 4)
    # Total rank: 49 + 16 + 16 + 4 = 85
    
    # First, define the rank-49 decomposition for 4x4x4 matrix multiplication
    # using the standard Strassen-like construction.
    
    # The 2x2 Strassen tensor has rank 7. We can use Kronecker products to build
    # the 4x4x4 tensor. The 4x4x4 tensor can be viewed as a Kronecker product of
    # two 2x2x2 tensors, but that gives rank 7*7=49, which is exactly what we need.
    
    # Define the 2x2 Strassen factors (each is 4x7)
    # U2: maps A (2x2) to intermediate
    # V2: maps B (2x2) to intermediate  
    # W2: maps intermediate to C (2x2)
    
    U2 = [
        [1, 0, 0, 1, 0, -1, 0],   # row 0: A[0,0]
        [0, 0, 1, 1, 0, 0, 0],    # row 1: A[0,1]
        [0, 1, 0, 0, 0, 1, 0],    # row 2: A[1,0]
        [1, 0, 0, 0, 0, 0, -1]    # row 3: A[1,1]
    ]
    
    V2 = [
        [0, 0, 1, 0, 0, 0, 1],    # row 0: B[0,0]
        [0, 1, 0, 0, 0, 0, 0],    # row 1: B[0,1]
        [0, 0, 0, 1, 0, 1, 0],    # row 2: B[1,0]
        [0, 0, 0, 0, 1, 0, -1]    # row 3: B[1,1]
    ]
    
    W2 = [
        [1, 0, 0, 0, 1, 0, 0],    # row 0: C[0,0]
        [0, 0, 0, 1, 0, 0, 1],    # row 1: C[0,1]
        [0, 0, 0, 0, 0, 1, 0],    # row 2: C[1,0]
        [0, 1, 0, 0, 0, 0, -1]    # row 3: C[1,1]
    ]
    
    # Verify: C[i,k] = sum_j A[i,j] B[j,k]
    # Using Kronecker product structure for 4x4x4:
    # Index mapping: i = 2*i1 + i2, j = 2*j1 + j2, k = 2*k1 + k2
    # The 4x4x4 tensor is the Kronecker product of two 2x2x2 tensors.
    # So U4[i,j] = U2[i1,j1] * U2[i2,j2] where i=2*i1+i2, j=2*j1+j2
    # Similarly for V4 and W4.
    
    # Build U4, V4, W4 (each 16x49)
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    for a in range(7):
        for b in range(7):
            col = a*7 + b
            for i1 in range(2):
                for i2 in range(2):
                    i = 2*i1 + i2
                    U4[i*4 + 2*i1 + i2][col] = U2[i1*2 + i2][a] * U2[i1*2 + i2][b]
            for j1 in range(2):
                for j2 in range(2):
                    j = 2*j1 + j2
                    V4[j*4 + 2*j1 + j2][col] = V2[j1*2 + j2][a] * V2[j1*2 + j2][b]
            for k1 in range(2):
                for k2 in range(2):
                    k = 2*k1 + k2
                    W4[k*4 + 2*k1 + k2][col] = W2[k1*2 + k2][a] * W2[k1*2 + k2][b]
    
    # Total rank: 49 (4x4x4) + 16 (4x4x1) + 16 (4x1x4) + 4 (4x1x1) = 85
    total_rank = 85
    U = [[0]*total_rank for _ in range(n*m)]  # 20 rows (4*5)
    V = [[0]*total_rank for _ in range(m*p)]  # 25 rows (5*5)
    W = [[0]*total_rank for _ in range(p*n)]  # 20 rows (5*4)
    
    # Part 1: 4x4x4 (A[0:4,0:4] * B[0:4,0:4] -> C[0:4,0:4])
    # A indices: i in 0..3, j in 0..3 -> row i*5+j
    # B indices: j in 0..3, k in 0..3 -> row j*5+k
    # C indices: k in 0..3, i in 0..3 -> row k*4+i (since p=5, n=4, so W[k*n+i])
    for i in range(4):
        for j in range(4):
            for col in range(49):
                U[i*5 + j][col] = U4[i*4 + j][col]
    for j in range(4):
        for k in range(4):
            for col in range(49):
                V[j*5 + k][col] = V4[j*4 + k][col]
    for k in range(4):
        for i in range(4):
            for col in range(49):
                W[k*4 + i][col] = W4[k*4 + i][col]
    
    # Part 2: 4x4x1 (A[0:4,0:4] * B[0:4,4] -> C[0:4,4])
    # A[i,j] for i in 0..3, j in 0..3
    # B[j,4] for j in 0..3
    # C[i,4] for i in 0..3
    col = 49
    for i in range(4):
        for j in range(4):
            U[i*5 + j][col] = 1
            V[j*5 + 4][col] = 1
            W[4*4 + i][col] = 1  # k=4, i=i -> row 4*4+i
            col += 1
    
    # Part 3: 4x1x4 (A[0:4,4] * B[4,0:4] -> C[0:4,0:4])
    # A[i,4] for i in 0..3
    # B[4,k] for k in 0..3
    # C[i,k] for i in 0..3, k in 0..3
    for i in range(4):
        for k in range(4):
            U[i*5 + 4][col] = 1
            V[4*5 + k][col] = 1
            W[k*4 + i][col] = 1
            col += 1
    
    # Part 4: 4x1x1 (A[0:4,4] * B[4,4] -> C[0:4,4])
    # A[i,4] for i in 0..3
    # B[4,4]
    # C[i,4] for i in 0..3
    for i in range(4):
        U[i*5 + 4][col] = 1
        V[4*5 + 4][col] = 1
        W[4*4 + i][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
