import numpy as np
import time

START_TIME = time.time()
BUDGET_SECONDS = 145.0

def check_time():
    return time.time() - START_TIME < BUDGET_SECONDS

def construct():
    n, m, p = 3, 4, 5
    dim_u = n * m  # 12
    dim_v = m * p  # 20
    dim_w = p * n  # 15
    
    # We will construct a rank 53 decomposition based on Strassen 4x4x4 + 4x4x1
    # and restrict it to 3x4x5.
    
    # Strassen 2x2x2 decomposition
    # U_22: 4x7, V_22: 4x7, W_22: 4x7
    # Indices: u=0,1,2,3 (00,01,10,11), v=0,1,2,3, w=0,1,2,3
    
    def strassen_2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1
        
        # M3 = A00(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[0, 2] = 1; W[1, 2] = 1
        
        # M4 = A11(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[2, 3] = 1; W[3, 3] = 1
        
        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = 1; W[3, 4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[2, 5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[1, 6] = 1
        
        # Correct W for Strassen 2x2:
        # C00 = M1 - M2 + M3 + M4
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6 (Wait, standard Strassen W)
        
        # Let's use the standard Strassen W matrix:
        # W[0,0]=1, W[0,1]=-1, W[0,2]=1, W[0,3]=1
        # W[1,2]=1, W[1,4]=1
        # W[2,1]=1, W[2,3]=1
        # W[3,0]=1, W[3,1]=-1, W[3,2]=1, W[3,5]=1
        
        W = np.zeros((4, 7))
        W[0, 0] = 1; W[0, 1] = -1; W[0, 2] = 1; W[0, 3] = 1
        W[1, 2] = 1; W[1, 4] = 1
        W[2, 1] = 1; W[2, 3] = 1
        W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
        
        return U, V, W

    U2, V2, W2 = strassen_2x2()

    # Construct 4x4x4
    # A is 4x4 (16), B is 4x4 (16), C is 4x4 (16)
    # 7 blocks of 2x2x2
    U444 = np.zeros((16, 49))
    V444 = np.zeros((16, 49))
    W444 = np.zeros((16, 49))
    
    # Block mapping:
    # A blocks: A00 (0-3), A01 (4-7), A10 (8-11), A11 (12-15)
    # B blocks: B00 (0-3), B01 (4-7), B10 (8-11), B11 (12-15)
    # C blocks: C00 (0-3), C01 (4-7), C10 (8-11), C11 (12-15)
    
    # Strassen uses:
    # M1: (A00+A11)(B00+B11) -> C00, C11
    # M2: (A10+A11)B00 -> C10, C11
    # M3: A00(B01-B11) -> C00, C01
    # M4: A11(B10-B00) -> C00, C10
    # M5: (A00+A01)B11 -> C01, C11
    # M6: (A10-A00)(B00+B01) -> C11
    # M7: (A01-A11)(B10+B11) -> C00, C01
    
    # We will map the 2x2 Strassen U,V,W to the 4x4 blocks.
    
    # M1
    # U: A00 (0-3) + A11 (12-15)
    U444[0:4, 0:7] += U2
    U444[12:16, 0:7] += U2
    # V: B00 (0-3) + B11 (12-15)
    V444[0:4, 0:7] += V2
    V444[12:16, 0:7] += V2
    # W: C00 (0-3) + C11 (12-15)
    W444[0:4, 0:7] += W2
    W444[12:16, 0:7] += W2

    # M2
    # U: A10 (8-11) + A11 (12-15)
    U444[8:12, 7:14] += U2
    U444[12:16, 7:14] += U2
    # V: B00 (0-3)
    V444[0:4, 7:14] += V2
    # W: C10 (8-11) + C11 (12-15)
    W444[8:12, 7:14] += W2
    W444[12:16, 7:14] += W2

    # M3
    # U: A00 (0-3)
    U444[0:4, 14:21] += U2
    # V: B01 (4-7) - B11 (12-15)
    V444[4:8, 14:21] += V2
    V444[12:16, 14:21] -= V2
    # W: C00 (0-3) + C01 (4-7)
    W444[0:4, 14:21] += W2
    W444[4:8, 14:21] += W2

    # M4
    # U: A11 (12-15)
    U444[12:16, 21:28] += U2
    # V: B10 (8-11) - B00 (0-3)
    V444[8:12, 21:28] += V2
    V444[0:4, 21:28] -= V2
    # W: C00 (0-3) + C10 (8-11)
    W444[0:4, 21:28] += W2
    W444[8:12, 21:28] += W2

    # M5
    # U: A00 (0-3) + A01 (4-7)
    U444[0:4, 28:35] += U2
    U444[4:8, 28:35] += U2
    # V: B11 (12-15)
    V444[12:16, 28:35] += V2
    # W: C01 (4-7) + C11 (12-15)
    W444[4:8, 28:35] += W2
    W444[12:16, 28:35] += W2

    # M6
    # U: A10 (8-11) - A00 (0-3)
    U444[8:12, 35:42] += U2
    U444[0:4, 35:42] -= U2
    # V: B00 (0-3) + B01 (4-7)
    V444[0:4, 35:42] += V2
    V444[4:8, 35:42] += V2
    # W: C11 (12-15)
    W444[12:16, 35:42] += W2

    # M7
    # U: A01 (4-7) - A11 (12-15)
    U444[4:8, 42:49] += U2
    U444[12:16, 42:49] -= U2
    # V: B10 (8-11) + B11 (12-15)
    V444[8:12, 42:49] += V2
    V444[12:16, 42:49] += V2
    # W: C00 (0-3) + C01 (4-7)
    W444[0:4, 42:49] += W2
    W444[4:8, 42:49] += W2

    # Construct 4x4x1
    # A is 4x4 (16), B is 4x1 (4), C is 4x1 (4)
    # Rank 4 (simple multiplication)
    U441 = np.zeros((16, 4))
    V441 = np.zeros((4, 4))
    W441 = np.zeros((4, 4))
    
    # B is 4x1. Indices 0,1,2,3.
    # A is 4x4.
    # C is 4x1.
    # C_i = sum_j A_ij B_j
    # For each j, we have a term A[:,j] * B[j] * C[:,j]?
    # No, C_i = sum_j A_ij B_j.
    # Tensor T_{u,v,w}. u=ij, v=j, w=i.
    # T_{ij, j, i} = 1.
    # For j=0: u=0,4,8,12. v=0. w=0,1,2,3.
    # This is not rank 1.
    # Rank of 4x4x1 is 4?
    # C = A B. B is 4x1.
    # C_i = A_i0 B_0 + A_i1 B_1 + A_i2 B_2 + A_i3 B_3.
    # This is a sum of 4 rank-1 terms?
    # Term j: A[:,j] (vector) * B[j] (scalar) * I (vector for C)?
    # No, C_i depends on A_ij.
    # T_{ij, j, i} = 1.
    # For fixed j, T_{ij, j, i} = 1 for all i.
    # This is a matrix of ones for i?
    # U_{ij, j} = 1. V_{j, j} = 1. W_{i, j} = 1.
    # Yes, rank 4.
    
    for j in range(4):
        for i in range(4):
            u = i * 4 + j
            v = j
            w = i
            U441[u, j] = 1
            V441[v, j] = 1
            W441[w, j] = 1

    # Combine 4x4x4 and 4x4x1 to 4x4x5
    # U445: 16 x 53
    # V445: 20 x 53 (B is 4x5, indices 0..19)
    #   B444 indices 0..15
    #   B441 indices 0..3 (mapped to 16..19? No, B441 is 4x1. B445 is 4x5.
    #   B445[j, k] = B444[j, k] for k<4, B441[j] for k=4?
    #   Actually, B441 is 4x1. Let's say it corresponds to the 5th column (k=4).
    #   So V445[0:16, 0:49] = V444
    #   V445[16:20, 49:53] = V441 (shifted)
    
    U445 = np.zeros((16, 53))
    V445 = np.zeros((20, 53))
    W445 = np.zeros((20, 53)) # C is 4x5, size 20
    
    U445[:, 0:49] = U444
    V445[0:16, 0:49] = V444
    W445[0:16, 0:49] = W444 # C444 is 4x4, indices 0..15 in C445 (k<4)
    
    U445[:, 49:53] = U441
    V445[16:20, 49:53] = V441
    W445[16:20, 49:53] = W441 # C441 is 4x1, indices 16..19 in C445 (k=4)

    # Restrict to 3x4x5
    # U345: 12 x 53
    # Keep rows of U445 where i < 3.
    # u = i*4 + j. i in 0..2.
    u_indices = [i * 4 + j for i in range(3) for j in range(4)]
    U345 = U445[u_indices, :]
    
    # V345: 20 x 53
    # Keep all rows of V445.
    V345 = V445
    
    # W345: 15 x 53
    # Keep rows of W445 where i < 3.
    # w = k*4 + i in W445.
    # We need w' = k*3 + i in W345.
    W345 = np.zeros((15, 53))
    for k in range(5):
        for i in range(3):
            w_idx_445 = k * 4 + i
            w_idx_345 = k * 3 + i
            W345[w_idx_345, :] = W445[w_idx_445, :]
    
    return {"U": U345.tolist(), "V": V345.tolist(), "W": W345.tolist()}
