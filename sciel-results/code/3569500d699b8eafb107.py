import numpy as np
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # 1. Strassen factors for 2x2x2
    # U2[i, j, r], V2[j, k, r], W2[k, i, r], r in 0..6
    U2 = np.zeros((2, 2, 7))
    V2 = np.zeros((2, 2, 7))
    W2 = np.zeros((2, 2, 7))
    
    # r=0: M1 = (A00+A11)(B00+B11)
    U2[0, 0, 0] = 1; U2[1, 1, 0] = 1
    V2[0, 0, 0] = 1; V2[1, 1, 0] = 1
    W2[0, 0, 0] = 1; W2[1, 1, 0] = 1
    
    # r=1: M2 = (A10+A11)B00
    U2[1, 0, 1] = 1; U2[1, 1, 1] = 1
    V2[0, 0, 1] = 1
    W2[0, 1, 1] = 1; W2[1, 1, 1] = -1
    
    # r=2: M3 = A00(B01-B11)
    U2[0, 0, 2] = 1
    V2[0, 1, 2] = 1; V2[1, 1, 2] = -1
    W2[1, 0, 2] = 1; W2[1, 1, 2] = 1
    
    # r=3: M4 = A11(B10-B00)
    U2[1, 1, 3] = 1
    V2[1, 0, 3] = 1; V2[0, 0, 3] = -1
    W2[0, 0, 3] = 1; W2[0, 1, 3] = 1
    
    # r=4: M5 = (A00+A01)B11
    U2[0, 0, 4] = 1; U2[0, 1, 4] = 1
    V2[1, 1, 4] = 1
    W2[0, 0, 4] = -1; W2[1, 0, 4] = 1
    
    # r=5: M6 = (A10-A00)(B00+B01)
    U2[1, 0, 5] = 1; U2[0, 0, 5] = -1
    V2[0, 0, 5] = 1; V2[0, 1, 5] = 1
    W2[1, 1, 5] = 1
    
    # r=6: M7 = (A01-A11)(B10+B11)
    U2[0, 1, 6] = 1; U2[1, 1, 6] = -1
    V2[1, 0, 6] = 1; V2[1, 1, 6] = 1
    W2[0, 0, 6] = 1

    # 2. Build 4x4x4 factors using Kronecker recursion
    # U4[i, j, R], V4[j, k, R], W4[k, i, R]
    # R = 7*s + r
    U4 = np.zeros((4, 4, 49))
    V4 = np.zeros((4, 4, 49))
    W4 = np.zeros((4, 4, 49))
    
    # Vectorized construction
    for i1 in range(2):
        for j1 in range(2):
            for i0 in range(2):
                for j0 in range(2):
                    i = 2*i1 + i0
                    j = 2*j1 + j0
                    for s in range(7):
                        for r in range(7):
                            R = 7*s + r
                            U4[i, j, R] = U2[i1, j1, s] * U2[i0, j0, r]
    
    for j1 in range(2):
        for k1 in range(2):
            for j0 in range(2):
                for k0 in range(2):
                    j = 2*j1 + j0
                    k = 2*k1 + k0
                    for s in range(7):
                        for r in range(7):
                            R = 7*s + r
                            V4[j, k, R] = V2[j1, k1, s] * V2[j0, k0, r]
    
    for k1 in range(2):
        for i1 in range(2):
            for k0 in range(2):
                for i0 in range(2):
                    k = 2*k1 + k0
                    i = 2*i1 + i0
                    for s in range(7):
                        for r in range(7):
                            R = 7*s + r
                            W4[k, i, R] = W2[k1, i1, s] * W2[k0, i0, r]

    # 3. Build 4x4x8 factors
    # U shape (16, rank), V shape (32, rank), W shape (32, rank)
    # U[i*m+j, R], V[j*p+k, R], W[k*n+i, R]
    
    # The rank is 49, not 98, because we can share the U and W factors
    # for the two 4x4 blocks of B.
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    # U is the same as U4, reshaped
    # U[i*4 + j, R] = U4[i, j, R]
    for i in range(4):
        for j in range(4):
            U[i*4 + j, :] = U4[i, j, :]
            
    # V is formed by stacking V4 for B1 and V4 for B2
    # V[j*8 + k, R]
    # For k in 0..3 (B1): V[j*8 + k, R] = V4[j, k, R]
    # For k in 4..7 (B2): V[j*8 + k, R] = V4[j, k-4, R]
    for j in range(4):
        for k in range(4):
            V[j*8 + k, :] = V4[j, k, :]
            V[j*8 + (k + 4), :] = V4[j, k, :]
            
    # W is formed by stacking W4 for C1 and W4 for C2
    # W[k*4 + i, R]
    # For k in 0..3 (C1): W[k*4 + i, R] = W4[k, i, R]
    # For k in 4..7 (C2): W[k*4 + i, R] = W4[k-4, i, R]
    for k in range(4):
        for i in range(4):
            W[k*4 + i, :] = W4[k, i, :]
            W[(k + 4)*4 + i, :] = W4[k, i, :]

    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
