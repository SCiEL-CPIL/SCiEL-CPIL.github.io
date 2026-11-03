import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    def verify(U, V, W, rank):
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # 1. Construct 4x4 * 4x4 Strassen decomposition (Rank 49)
    # U2, V2, W2 for 2x2 * 2x2
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    # M2 = (A21+A22)B11
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = 1
    # M3 = A11(B21-B22)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[0, 2] = 1; W2[1, 2] = 1
    # M4 = (A11-A22)(B12+B22)
    U2[0, 3] = 1; U2[3, 3] = -1
    V2[1, 3] = 1; V2[3, 3] = 1
    W2[0, 3] = 1; W2[3, 3] = 1
    # M5 = (A11+A12)B22
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[1, 4] = 1; W2[3, 4] = 1
    # M6 = (A21-A11)(B11+B12)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[2, 5] = 1; W2[0, 5] = 1
    # M7 = (A12-A22)(B21+B22)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[1, 6] = 1; W2[2, 6] = 1

    # Build 4x4 * 4x4 using Kronecker product of Strassen
    U44 = np.zeros((16, 49))
    V44 = np.zeros((16, 49))
    W44 = np.zeros((16, 49))
    
    for t1 in range(7):
        for t2 in range(7):
            col = t1 * 7 + t2
            for i1 in range(2):
                if U2[i1, t1] != 0:
                    for i2 in range(2):
                        if U2[i2, t2] != 0:
                            row = i1 * 2 + i2
                            U44[row, col] = U2[i1, t1] * U2[i2, t2]
            for i1 in range(2):
                if V2[i1, t1] != 0:
                    for i2 in range(2):
                        if V2[i2, t2] != 0:
                            row = i1 * 2 + i2
                            V44[row, col] = V2[i1, t1] * V2[i2, t2]
            for i1 in range(2):
                if W2[i1, t1] != 0:
                    for i2 in range(2):
                        if W2[i2, t2] != 0:
                            row = i1 * 2 + i2
                            W44[row, col] = W2[i1, t1] * W2[i2, t2]

    # 2. Construct 4x4 * 4x1 decomposition (Rank 4)
    U41 = np.zeros((16, 4))
    V41 = np.zeros((20, 4))
    W41 = np.zeros((20, 4))
    
    for r in range(4):
        for i in range(4):
            for j in range(4):
                if i == r:
                    U41[i*4+j, r] = 1.0
                    W41[16+i, r] = 1.0
        for j in range(4):
            V41[j*5+4, r] = 1.0

    # 3. Combine into full 4x4 * 4x5 decomposition
    rank = 53
    U = np.zeros((16, rank))
    V = np.zeros((20, rank))
    W = np.zeros((20, rank))
    
    # Fill 4x4 * 4x4 part
    U[:, 0:49] = U44
    for j in range(4):
        for k in range(4):
            V[j*5+k, 0:49] = V44[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[k*4+i, 0:49] = W44[k*4+i, 0:49]
            
    # Fill 4x4 * 4x1 part
    U[:, 49:53] = U41
    V[:, 49:53] = V41
    W[:, 49:53] = W41

    # Verify
    if not verify(U, V, W, rank):
        # Fallback to schoolbook if Strassen construction failed
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
