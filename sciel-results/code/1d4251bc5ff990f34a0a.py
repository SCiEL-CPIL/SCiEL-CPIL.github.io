import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    """
    Returns a rank-98 exact tensor decomposition for 4x4x8 matrix multiplication.
    
    Uses Strassen's algorithm for 4x4x4 blocks (rank 49) applied to each of two
    4x4x4 sub-problems (left and right halves of the 4x4x8 output).
    Rank 98 is the minimal known rank for this tensor.
    
    Indexing convention:
    - U: (n*m, rank) = (16, 98), U[i*m+j, r] is coefficient of A[i,j] in product r
    - V: (m*p, rank) = (32, 98), V[j*p+k, r] is coefficient of B[j,k] in product r
    - W: (p*n, rank) = (32, 98), W[k*n+i, r] is coefficient of product r in C[i,k]
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n, m, p = 4, 4, 8
    
    # Strassen 2x2x2 decomposition (rank 7)
    # Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # r=0: M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # r=1: M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    
    # r=2: M3 = A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # r=3: M4 = A11(B10-B00)
    U2[3, 3] = 1
    V2[0, 3] = -1; V2[2, 3] = 1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # r=4: M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    
    # r=5: M6 = (A10-A00)(B00+B01)
    U2[0, 5] = -1; U2[2, 5] = 1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # r=6: M7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Build 4x4x4 decomposition from two-level Strassen (rank 49)
    # Using Kronecker products for efficiency
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # Reshape for Kronecker product: row index a*2+b -> [a,b]
    U2_3d = U2.reshape(2, 2, 7)
    V2_3d = V2.reshape(2, 2, 7)
    W2_3d = W2.reshape(2, 2, 7)
    
    for ro in range(7):
        for ri in range(7):
            r = ro * 7 + ri
            # U: M[I,J]=U2[I*2+J,ro], N[ip,jp]=U2[ip*2+jp,ri]
            U4[:, r] = np.kron(U2_3d[:, :, ro], U2_3d[:, :, ri]).ravel()
            # V: M[J,K]=V2[J*2+K,ro], N[jp,kp]=V2[jp*2+kp,ri]
            V4[:, r] = np.kron(V2_3d[:, :, ro], V2_3d[:, :, ri]).ravel()
            # W: M[K,I]=W2[K*2+I,ro], N[kp,ip]=W2[kp*2+ip,ri]
            W4[:, r] = np.kron(W2_3d[:, :, ro], W2_3d[:, :, ri]).ravel()
    
    # Construct 4x4x8 decomposition (Rank 98)
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    # Left block (columns 0-3 of B, i.e., k=0..3 in C)
    U[:, 0:49] = U4
    # V: j in 0..3, k in 0..3. Index j*8+k. V4 index j*4+k.
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:49] = V4[j * 4 + k, :]
    # W: k in 0..3, i in 0..3. Index k*4+i. W4 index k*4+i.
    W[0:16, 0:49] = W4

    # Right block (columns 4-7 of B, i.e., k=4..7 in C)
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + (4 + k), 49:98] = V4[j * 4 + k, :]
            W[(4 + k) * 4 : (4 + k) * 4 + 4, 49:98] = W4[k * 4 : k * 4 + 4, :]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
