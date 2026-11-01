"""
Rank-minimal exact tensor decomposition for 2x2 matrix multiplication (Strassen, rank 7).
The minimal rank for 2x2 matrix multiplication over the reals is 7.
We return Strassen's decomposition verified numerically.
"""
import time

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Return Strassen's rank-7 tensor decomposition for 2x2 matrix multiplication.
    
    Interface:
    - U: shape (n*m, rank) = (4, 7), entries U[i*m+j, c]
    - V: shape (m*p, rank) = (4, 7), entries V[j*p+k, c]
    - W: shape (p*n, rank) = (4, 7), entries W[k*n+i, c]
    
    Reconstruction:
    For each c, compute scalar m_c = (sum_{i,j} U[i*2+j, c] * A[i,j]) * (sum_{j,k} V[j*2+k, c] * B[j,k])
    Then C[i,k] = sum_c W[k*2+i, c] * m_c
    """
    import numpy as np
    
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # c=0: m1 = (a11+a22)(b11+b22)
    # A-side: A[0,0] + A[1,1]
    U[0, 0] = 1  # A[0,0]
    U[3, 0] = 1  # A[1,1]
    # B-side: B[0,0] + B[1,1]
    V[0, 0] = 1  # B[0,0]
    V[3, 0] = 1  # B[1,1]
    # C-side: contributes to C[0,0] and C[1,1]
    W[0, 0] = 1  # C[0,0] (k=0, i=0)
    W[3, 0] = 1  # C[1,1] (k=1, i=1)
    
    # c=1: m2 = (a21+a22)b11
    # A-side: A[1,0] + A[1,1]
    U[2, 1] = 1  # A[1,0]
    U[3, 1] = 1  # A[1,1]
    # B-side: B[0,0]
    V[0, 1] = 1  # B[0,0]
    # C-side: contributes to C[1,0] and C[1,1]
    W[1, 1] = 1   # C[1,0] (k=0, i=1)
    W[3, 1] = -1  # C[1,1] (k=1, i=1)
    
    # c=2: m3 = a11(b12-b22)
    # A-side: A[0,0]
    U[0, 2] = 1  # A[0,0]
    # B-side: B[0,1] - B[1,1]
    V[1, 2] = 1   # B[0,1]
    V[3, 2] = -1  # B[1,1]
    # C-side: contributes to C[0,1] and C[1,1]
    W[2, 2] = 1  # C[0,1] (k=1, i=0)
    W[3, 2] = 1  # C[1,1] (k=1, i=1)
    
    # c=3: m4 = a22(b21-b11)
    # A-side: A[1,1]
    U[3, 3] = 1  # A[1,1]
    # B-side: B[1,0] - B[0,0]
    V[2, 3] = 1   # B[1,0]
    V[0, 3] = -1  # B[0,0]
    # C-side: contributes to C[0,0] and C[1,0]
    W[0, 3] = 1  # C[0,0] (k=0, i=0)
    W[1, 3] = 1  # C[1,0] (k=0, i=1)
    
    # c=4: m5 = (a11+a12)b22
    # A-side: A[0,0] + A[0,1]
    U[0, 4] = 1  # A[0,0]
    U[1, 4] = 1  # A[0,1]
    # B-side: B[1,1]
    V[3, 4] = 1  # B[1,1]
    # C-side: contributes to C[0,0] and C[0,1]
    W[0, 4] = -1  # C[0,0] (k=0, i=0)
    W[2, 4] = 1   # C[0,1] (k=1, i=0)
    
    # c=5: m6 = (a21-a11)(b11+b12)
    # A-side: A[1,0] - A[0,0]
    U[2, 5] = 1   # A[1,0]
    U[0, 5] = -1  # A[0,0]
    # B-side: B[0,0] + B[0,1]
    V[0, 5] = 1  # B[0,0]
    V[1, 5] = 1  # B[0,1]
    # C-side: contributes to C[1,1]
    W[3, 5] = 1  # C[1,1] (k=1, i=1)
    
    # c=6: m7 = (a12-a22)(b21+b22)
    # A-side: A[0,1] - A[1,1]
    U[1, 6] = 1   # A[0,1]
    U[3, 6] = -1  # A[1,1]
    # B-side: B[1,0] + B[1,1]
    V[2, 6] = 1  # B[1,0]
    V[3, 6] = 1  # B[1,1]
    # C-side: contributes to C[0,0]
    W[0, 6] = 1  # C[0,0] (k=0, i=0)
    
    # Verify numerically with random integer matrices
    for trial in range(50):
        A = np.random.randint(-10, 10, (2, 2)).astype(float)
        B = np.random.randint(-10, 10, (2, 2)).astype(float)
        C_expected = A @ B
        
        A_flat = A.flatten()  # [A00, A01, A10, A11]
        B_flat = B.flatten()  # [B00, B01, B10, B11]
        
        # Compute m_c for each c
        m = np.zeros(rank)
        for c in range(rank):
            a_c = np.dot(U[:, c], A_flat)
            b_c = np.dot(V[:, c], B_flat)
            m[c] = a_c * b_c
        
        # Compute C from decomposition
        C_recon = np.zeros((2, 2))
        for i in range(2):
            for k in range(2):
                C_recon[i, k] = np.dot(W[k * 2 + i, :], m)
        
        if not np.allclose(C_recon, C_expected, atol=1e-10):
            raise ValueError(f"Decomposition failed on trial {trial}")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
