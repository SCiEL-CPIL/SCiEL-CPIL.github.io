import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n, m, p = 4, 4, 8
    
    # Strassen 2x2x2 rank-7 decomposition
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    
    # Build 4x4x4 using Kronecker product (rank 49)
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    # Correct Kronecker product construction
    # Rank of 4x4x4 is 7*7 = 49
    # Index r in [0, 49) maps to (r1, r2) where r = r1 * 7 + r2
    for i in range(4):
        for j in range(4):
            i1, i2 = divmod(i, 2)
            j1, j2 = divmod(j, 2)
            for r in range(49):
                r1, r2 = divmod(r, 7)
                U4[i*4+j, r] = U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
    
    for j in range(4):
        for k in range(4):
            j1, j2 = divmod(j, 2)
            k1, k2 = divmod(k, 2)
            for r in range(49):
                r1, r2 = divmod(r, 7)
                V4[j*4+k, r] = V2[j1*2+k1, r1] * V2[j2*2+k2, r2]
    
    for k in range(4):
        for i in range(4):
            k1, k2 = divmod(k, 2)
            i1, i2 = divmod(i, 2)
            for r in range(49):
                r1, r2 = divmod(r, 7)
                W4[k*4+i, r] = W2[k1*2+i1, r1] * W2[k2*2+i2, r2]
    
    # Build 4x4x8: two independent 4x4x4 blocks (rank 98)
    rank = 98
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((32, rank), dtype=float)
    W = np.zeros((32, rank), dtype=float)
    
    # First half
    U[:, 0:49] = U4
    V[0:16, 0:49] = V4  # j=0..3, k=0..3 -> indices 0..15
    W[0:16, 0:49] = W4  # k=0..3, i=0..3 -> indices 0..15
    
    # Second half
    U[:, 49:98] = U4
    V[16:32, 49:98] = V4 # j=0..3, k=4..7 -> indices 16..31
    W[16:32, 49:98] = W4 # k=4..7, i=0..3 -> indices 16..31
    
    # Verify
    if time.time() < deadline:
        valid = True
        # Sample check for speed
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > 1e-6:
                        valid = False
                        break
                if not valid: break
            if not valid: break
        
        if not valid:
            # Fallback
            rank = 128
            U = np.zeros((16, rank), dtype=float)
            V = np.zeros((32, rank), dtype=float)
            W = np.zeros((32, rank), dtype=float)
            t = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j, t] = 1
                        V[j*p+k, t] = 1
                        W[k*n+i, t] = 1
                        t += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
