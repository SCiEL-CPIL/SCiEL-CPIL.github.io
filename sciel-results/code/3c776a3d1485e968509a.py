import numpy as np

def construct():
    """
    Return a rank-28 exact decomposition for 2x3 * 3x5 matrix multiplication.
    
    Strategy:
    - Split middle dimension m=3 into j=0 and j in {1,2}.
    - Block 1 (j=0): 2x1x5, rank 10 (schoolbook).
    - Block 2 (j in {1,2}): 2x2x5.
      - Split k into {0,1}, {2}, {3,4}.
      - Sub-block 2a (k=0,1): 2x2x2, rank 7 (Strassen).
      - Sub-block 2b (k=2): 2x2x1, rank 4 (schoolbook).
      - Sub-block 2c (k=3,4): 2x2x2, rank 7 (Strassen).
    - Total rank: 10 + 7 + 4 + 7 = 28.
    
    Indexing conventions:
    - A is 2x3, flattened row-major: A[i,j] -> index i*3+j
    - B is 3x5, flattened row-major: B[j,k] -> index j*5+k
    - C is 2x5, output follows W[k*n+i, r] convention: C[i,k] -> index k*2+i
    """
    n, m, p = 2, 3, 5
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 28
    V = [[0] * rank for _ in range(m * p)]  # 15 x 28
    W = [[0] * rank for _ in range(p * n)]  # 10 x 28
    
    col = 0
    
    # Block 1: j=0, all k in {0,1,2,3,4}
    # C[i,k] += A[i,0] * B[0,k]
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    # col is now 10
    
    # Block 2a: j in {1,2}, k in {0,1} — Strassen 2x2x2
    # Local indices: i' in {0,1}, j' in {0,1} (j=j'+1), k' in {0,1} (k=k')
    # U index: i * 3 + (j' + 1)
    # V index: (j' + 1) * 5 + k'
    # W index: k' * 2 + i
    
    # M1: (A[0,1]+A[1,2]) * (B[1,0]+B[2,1]) -> C[0,0] += 1, C[1,1] += 1
    U[0 * 3 + 1][col] = 1
    U[1 * 3 + 2][col] = 1
    V[1 * 5 + 0][col] = 1
    V[2 * 5 + 1][col] = 1
    W[0 * 2 + 0][col] = 1
    W[1 * 2 + 1][col] = 1
    col += 1
    
    # M2: (A[1,1]+A[1,2]) * B[1,0] -> C[1,0] += 1, C[1,1] += -1
    U[1 * 3 + 1][col] = 1
    U[1 * 3 + 2][col] = 1
    V[1 * 5 + 0][col] = 1
    W[0 * 2 + 1][col] = 1
    W[1 * 2 + 1][col] = -1
    col += 1
    
    # M3: A[0,1] * (B[1,1]-B[2,1]) -> C[0,1] += 1, C[1,1] += 1
    U[0 * 3 + 1][col] = 1
    V[1 * 5 + 1][col] = 1
    V[2 * 5 + 1][col] = -1
    W[1 * 2 + 0][col] = 1
    W[1 * 2 + 1][col] = 1
    col += 1
    
    # M4: A[1,2] * (B[2,0]-B[1,0]) -> C[0,0] += 1, C[1,0] += 1
    U[1 * 3 + 2][col] = 1
    V[2 * 5 + 0][col] = 1
    V[1 * 5 + 0][col] = -1
    W[0 * 2 + 0][col] = 1
    W[0 * 2 + 1][col] = 1
    col += 1
    
    # M5: (A[0,1]+A[0,2]) * B[2,1] -> C[0,0] += -1, C[0,1] += 1
    U[0 * 3 + 1][col] = 1
    U[0 * 3 + 2][col] = 1
    V[2 * 5 + 1][col] = 1
    W[0 * 2 + 0][col] = -1
    W[1 * 2 + 0][col] = 1
    col += 1
    
    # M6: (A[1,1]-A[0,1]) * (B[1,0]+B[1,1]) -> C[1,1] += 1
    U[1 * 3 + 1][col] = 1
    U[0 * 3 + 1][col] = -1
    V[1 * 5 + 0][col] = 1
    V[1 * 5 + 1][col] = 1
    W[1 * 2 + 1][col] = 1
    col += 1
    
    # M7: (A[0,2]-A[1,2]) * (B[2,0]+B[2,1]) -> C[0,0] += 1
    U[0 * 3 + 2][col] = 1
    U[1 * 3 + 2][col] = -1
    V[2 * 5 + 0][col] = 1
    V[2 * 5 + 1][col] = 1
    W[0 * 2 + 0][col] = 1
    col += 1
    
    # col is now 17
    
    # Block 2b: j in {1,2}, k=2 — schoolbook 2x2x1
    # C[i,2] += A[i,1]*B[1,2] + A[i,2]*B[2,2]
    for i in range(n):
        for j_local in range(2):
            j = j_local + 1
            U[i * 3 + j][col] = 1
            V[j * 5 + 2][col] = 1
            W[2 * 2 + i][col] = 1
            col += 1
    # col is now 21
    
    # Block 2c: j in {1,2}, k in {3,4} — Strassen 2x2x2
    # Local indices: i' in {0,1}, j' in {0,1} (j=j'+1), k' in {0,1} (k=k'+3)
    # U index: i * 3 + (j' + 1)
    # V index: (j' + 1) * 5 + (k' + 3)
    # W index: (k' + 3) * 2 + i
    
    # M1: (A[0,1]+A[1,2]) * (B[1,3]+B[2,4]) -> C[0,3] += 1, C[1,4] += 1
    U[0 * 3 + 1][col] = 1
    U[1 * 3 + 2][col] = 1
    V[1 * 5 + 3][col] = 1
    V[2 * 5 + 4][col] = 1
    W[3 * 2 + 0][col] = 1
    W[4 * 2 + 1][col] = 1
    col += 1
    
    # M2: (A[1,1]+A[1,2]) * B[1,3] -> C[1,3] += 1, C[1,4] += -1
    U[1 * 3 + 1][col] = 1
    U[1 * 3 + 2][col] = 1
    V[1 * 5 + 3][col] = 1
    W[3 * 2 + 1][col] = 1
    W[4 * 2 + 1][col] = -1
    col += 1
    
    # M3: A[0,1] * (B[1,4]-B[2,4]) -> C[0,4] += 1, C[1,4] += 1
    U[0 * 3 + 1][col] = 1
    V[1 * 5 + 4][col] = 1
    V[2 * 5 + 4][col] = -1
    W[4 * 2 + 0][col] = 1
    W[4 * 2 + 1][col] = 1
    col += 1
    
    # M4: A[1,2] * (B[2,3]-B[1,3]) -> C[0,3] += 1, C[1,3] += 1
    U[1 * 3 + 2][col] = 1
    V[2 * 5 + 3][col] = 1
    V[1 * 5 + 3][col] = -1
    W[3 * 2 + 0][col] = 1
    W[3 * 2 + 1][col] = 1
    col += 1
    
    # M5: (A[0,1]+A[0,2]) * B[2,4] -> C[0,3] += -1, C[0,4] += 1
    U[0 * 3 + 1][col] = 1
    U[0 * 3 + 2][col] = 1
    V[2 * 5 + 4][col] = 1
    W[3 * 2 + 0][col] = -1
    W[4 * 2 + 0][col] = 1
    col += 1
    
    # M6: (A[1,1]-A[0,1]) * (B[1,3]+B[1,4]) -> C[1,4] += 1
    U[1 * 3 + 1][col] = 1
    U[0 * 3 + 1][col] = -1
    V[1 * 5 + 3][col] = 1
    V[1 * 5 + 4][col] = 1
    W[4 * 2 + 1][col] = 1
    col += 1
    
    # M7: (A[0,2]-A[1,2]) * (B[2,3]+B[2,4]) -> C[0,3] += 1
    U[0 * 3 + 2][col] = 1
    U[1 * 3 + 2][col] = -1
    V[2 * 5 + 3][col] = 1
    V[2 * 5 + 4][col] = 1
    W[3 * 2 + 0][col] = 1
    col += 1
    
    # col should be 28
    
    return {"U": U, "V": V, "W": W}
