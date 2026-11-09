import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe
    
    n, m, p = DIMENSIONS
    
    # We construct a decomposition of rank 36.
    # Based on R(3, 2, 2) * R(1, 2, 3) = 6 * 6 = 36.
    # T1: 3x2x2 (Rank 6)
    # T2: 1x2x3 (Rank 6)
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # T1: 3x2x2
    # A: 3x2, B: 2x2, C: 3x2
    # Decomposed into 3 independent 1x2x2 multiplications.
    # Each 1x2x2 has rank 2.
    # U1 shape: (3*2) x 6 = 6x6
    # V1 shape: (2*2) x 6 = 4x6
    # W1 shape: (2*3) x 6 = 6x6  (p*n = 2*3=6)
    
    # T2: 1x2x3
    # A: 1x2, B: 2x3, C: 1x3
    # Decomposed into 3 independent 1x2x1 multiplications? No, 3 dot products of length 2.
    # Each 1x2x1 has rank 2? No, 1x2x1 is just a_1 b_1 + a_2 b_2. Rank 2.
    # So T2 rank is 3 * 2 = 6.
    # U2 shape: (1*2) x 6 = 2x6
    # V2 shape: (2*3) x 6 = 6x6
    # W2 shape: (3*1) x 6 = 3x6
    
    # Kronecker product of factors
    # U = U1 x U2 ? No, U indices are (n*m, rank).
    # n*m = 3*4 = 12.
    # U1 rows: 3*2=6. U2 rows: 1*2=2.
    # Total rows 6*2=12. Correct.
    
    # V indices are (m*p, rank).
    # m*p = 4*6 = 24.
    # V1 rows: 2*2=4. V2 rows: 2*3=6.
    # Total rows 4*6=24. Correct.
    
    # W indices are (p*n, rank).
    # p*n = 6*3 = 18.
    # W1 rows: 2*3=6. W2 rows: 3*1=3.
    # Total rows 6*3=18. Correct.
    
    # Construct T1 factors (Rank 6)
    U1 = [[0]*6 for _ in range(6)]
    V1 = [[0]*6 for _ in range(4)]
    W1 = [[0]*6 for _ in range(6)]
    
    for i in range(3):
        for c in range(2):
            col = i*2 + c
            # U1 row index: i*2 + j (j in 0,1)
            # For c=0: U[0]=1, U[1]=0
            # For c=1: U[0]=0, U[1]=1
            U1[i*2 + c][col] = 1
            
            # V1 row index: j*2 + k (j in 0,1, k in 0,1)
            # For c=0: V[0]=1 (j=0, k=0), V[1]=0, V[2]=0, V[3]=0
            # For c=1: V[0]=0, V[1]=0, V[2]=1 (j=1, k=0), V[3]=0
            if c == 0:
                V1[0][col] = 1
            else:
                V1[2][col] = 1
            
            # W1 row index: k*3 + i (k in 0,1)
            # For c=0: W[0]=1 (k=0, i), W[1]=0
            # For c=1: W[0]=0, W[1]=1 (k=1, i)
            if c == 0:
                W1[i][col] = 1  # k=0 -> index i
            else:
                W1[3+i][col] = 1 # k=1 -> index 3+i

    # Construct T2 factors (Rank 6)
    U2 = [[0]*6 for _ in range(2)]
    V2 = [[0]*6 for _ in range(6)]
    W2 = [[0]*6 for _ in range(3)]
    
    for k in range(3):
        for c in range(2):
            col = k*2 + c
            # U2 row index: j (j in 0,1)
            if c == 0:
                U2[0][col] = 1
            else:
                U2[1][col] = 1
            
            # V2 row index: j*3 + k (j in 0,1)
            if c == 0:
                V2[k][col] = 1 # j=0 -> index k
            else:
                V2[3+k][col] = 1 # j=1 -> index 3+k
            
            # W2 row index: k (k in 0,1,2)
            W2[k][col] = 1

    # Combine U1, U2 to U
    # U row (i1*2 + j1, i2*2 + j2) ?
    # n1=3, m1=2. n2=1, m2=2.
    # U1 row: i1*m1 + j1 = i1*2 + j1.
    # U2 row: i2*m2 + j2 = i2*2 + j2.
    # U row: (i1*n2 + i2) * (m1*m2) + j1*m2 + j2 ?
    # Standard Kronecker:
    # Row index in U: (i1, j1, i2, j2) -> (i1*n2 + i2) * (m1*m2) + j1*m2 + j2
    # n = n1*n2 = 3. m = m1*m2 = 4.
    # U row = i*m + j = (i1*n2 + i2)*4 + (j1*m2 + j2)
    # i1 in 0..2, i2 in 0..0.
    # j1 in 0..1, j2 in 0..1.
    
    for r1 in range(6):
        for r2 in range(2):
            # Map r1, r2 to i1, j1, i2, j2
            i1 = r1 // 2
            j1 = r1 % 2
            i2 = r2 // 2
            j2 = r2 % 2
            
            u_row = (i1 * 1 + i2) * 4 + (j1 * 2 + j2)
            
            for c in range(36):
                c1 = c // 6
                c2 = c % 6
                U[u_row][c] = U1[r1][c1] * U2[r2][c2]
    
    # Combine V1, V2 to V
    # V row: (j1, k1, j2, k2) -> (j1*m2 + j2) * (p1*p2) + k1*p2 + k2
    # m = 4, p = 6.
    # V row = j*p + k = (j1*2 + j2)*6 + (k1*3 + k2)
    
    for r1 in range(4):
        for r2 in range(6):
            j1 = r1 // 2
            k1 = r1 % 2
            j2 = r2 // 3
            k2 = r2 % 3
            
            v_row = (j1 * 2 + j2) * 6 + (k1 * 3 + k2)
            
            for c in range(36):
                c1 = c // 6
                c2 = c % 6
                V[v_row][c] = V1[r1][c1] * V2[r2][c2]
    
    # Combine W1, W2 to W
    # W row: (k1, i1, k2, i2) -> (k1*p2 + k2) * (n1*n2) + i1*n2 + i2
    # p = 6, n = 3.
    # W row = k*n + i = (k1*3 + k2)*3 + (i1*1 + i2)
    
    for r1 in range(6):
        for r2 in range(3):
            k1 = r1 // 3
            i1 = r1 % 3
            k2 = r2 // 1
            i2 = r2 % 1
            
            w_row = (k1 * 3 + k2) * 3 + (i1 * 1 + i2)
            
            for c in range(36):
                c1 = c // 6
                c2 = c % 6
                W[w_row][c] = W1[r1][c1] * W2[r2][c2]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
