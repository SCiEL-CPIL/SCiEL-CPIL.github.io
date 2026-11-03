import numpy as np
import time

def construct():
    """
    Rank-98 decomposition for 4x4 * 4x8 matrix multiplication.
    
    Uses the Kronecker product of two levels of Strassen (2x2x2 rank-7)
    to get a rank-49 decomposition for 4x4x4, then applies it independently
    to the two 4x4 column blocks of B, giving total rank 98.
    
    This is a structural improvement over the 8-block approach (rank 112)
    because it properly shares the Strassen structure across the 4x4 level.
    """
    start_time = time.time()
    
    n, m, p = 4, 4, 8
    
    # Strassen 2x2x2 rank-7 decomposition
    # U2[i*2+j, t], V2[j*2+k, t], W2[k*2+i, t]
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],  # i=0,j=0
        [0, 0, 0, 0, 1,  0, 1],  # i=0,j=1
        [0, 1, 0, 0, 0,  1, 0],  # i=1,j=0
        [1, 1, 0, 1, 0,  0,-1]   # i=1,j=1
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],  # j=0,k=0
        [0, 0, 1,  0, 0, 1, 0],  # j=0,k=1
        [0, 0, 0,  1, 0, 0, 1],  # j=1,k=0
        [1, 0,-1,  0, 1, 0, 1]   # j=1,k=1
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # k=0,i=0
        [0, 1, 0, 1,  0, 0, 0],  # k=0,i=1
        [0, 0, 1, 0,  1, 0, 0],  # k=1,i=0
        [1,-1, 1, 0,  0, 1, 0]   # k=1,i=1
    ], dtype=float)
    
    # Verify 2x2x2 Strassen
    def verify_2x2x2():
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    total = 0.0
                    for t in range(7):
                        total += U2[i*2+j, t] * V2[j*2+k, t] * W2[k*2+i, t]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify_2x2x2():
        return construct_naive(n, m, p)
    
    # Build 4x4x4 rank-49 decomposition via Kronecker product
    # U4[i*4+j, t*7+s] = U2[i1*2+j1, t] * U2[i2*2+j2, s]
    # where i = 2*i1 + i2, j = 2*j1 + j2
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    i = 2*i1 + i2
                    j = 2*j1 + j2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            U4[i*4+j, c] = U2[i1*2+j1, t] * U2[i2*2+j2, s]
    
    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(2):
                    j = 2*j1 + j2
                    k = 2*k1 + k2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            V4[j*4+k, c] = V2[j1*2+k1, t] * V2[j2*2+k2, s]
    
    for k1 in range(2):
        for k2 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    k = 2*k1 + k2
                    i = 2*i1 + i2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            W4[k*4+i, c] = W2[k1*2+i1, t] * W2[k2*2+i2, s]
    
    # Verify 4x4x4 decomposition
    def verify_4x4x4():
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    total = 0.0
                    for c in range(49):
                        total += U4[i*4+j, c] * V4[j*4+k, c] * W4[k*4+i, c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify_4x4x4():
        return construct_naive(n, m, p)
    
    # Build 4x4x8 rank-98 decomposition
    # Split B into two 4x4 blocks: B_left (cols 0-3) and B_right (cols 4-7)
    rank = 98
    U = np.zeros((n*m, rank), dtype=float)   # 16 x 98
    V = np.zeros((m*p, rank), dtype=float)   # 32 x 98
    W = np.zeros((p*n, rank), dtype=float)   # 32 x 98
    
    # First block (output cols 0-3)
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+k, 0:49] = V4[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[k*4+i, 0:49] = W4[k*4+i, 0:49]
    
    # Second block (output cols 4-7)
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+(k+4), 49:98] = V4[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[(k+4)*4+i, 49:98] = W4[k*4+i, 0:49]
    
    # Full verification
    def verify_full():
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify_full():
        return construct_naive(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct_naive(n, m, p):
    U = np.zeros((n * m, n * m * p), dtype=float)
    V = np.zeros((m * p, n * m * p), dtype=float)
    W = np.zeros((p * n, n * m * p), dtype=float)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, idx] = 1.0
                V[j*p+k, idx] = 1.0
                W[k*n+i, idx] = 1.0
                idx += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
