import numpy as np

def construct():
    """
    Construct a rank-minimal tensor decomposition for multiplying a 2x3 matrix by a 3x8 matrix.
    
    Strategy:
    - Split the middle dimension (size 3) into a 1-dim part and a 2-dim part.
    - The 1-dim part (2x1 * 1x8) has rank 16.
    - The 2-dim part (2x2 * 2x8) is 4 independent 2x2 * 2x2 multiplications.
    - Each 2x2 * 2x2 multiplication uses Strassen's algorithm (rank 7).
    - Total rank: 16 + 4 * 7 = 44.
    
    Index conventions:
    - U has shape (n*m, rank) = (6, rank): index a = i*m + j
    - V has shape (m*p, rank) = (24, rank): index b = j*p + k
    - W has shape (p*n, rank) = (16, rank): index c = k*n + i
    """
    n, m, p = 2, 3, 8
    rank = 44
    
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    
    col = 0
    
    # Part 1: Middle index j=0
    # A[:, 0] * B[0, :] -> 2x1 * 1x8
    # This is an outer product of a 2-vector and an 8-vector.
    # Rank 16.
    for i in range(n):
        for k in range(p):
            U[i*m + 0, col] = 1
            V[0*p + k, col] = 1
            W[k*n + i, col] = 1
            col += 1
            
    # Part 2: Middle indices j=1, 2
    # A[:, 1:3] * B[1:3, :] -> 2x2 * 2x8
    # Split into 4 blocks of 2 columns each.
    for block in range(4):
        k0 = block * 2
        k1 = block * 2 + 1
        
        # Strassen for 2x2 * 2x2
        # A_sub = [[A[0,1], A[0,2]], [A[1,1], A[1,2]]]
        # B_sub = [[B[1,k0], B[1,k1]], [B[2,k0], B[2,k1]]]
        
        # M1 = (A[0,1] + A[1,2]) * (B[1,k0] + B[2,k1])
        U[0*m + 1, col] = 1
        U[1*m + 2, col] = 1
        V[1*p + k0, col] = 1
        V[2*p + k1, col] = 1
        W[k0*n + 0, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        
        # M2 = (A[1,1] + A[1,2]) * B[1,k0]
        U[1*m + 1, col] = 1
        U[1*m + 2, col] = 1
        V[1*p + k0, col] = 1
        W[k0*n + 1, col] = 1
        W[k1*n + 1, col] = -1
        col += 1
        
        # M3 = A[0,1] * (B[1,k1] - B[2,k1])
        U[0*m + 1, col] = 1
        V[1*p + k1, col] = 1
        V[2*p + k1, col] = -1
        W[k1*n + 0, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        
        # M4 = A[1,2] * (B[2,k0] - B[1,k0])
        U[1*m + 2, col] = 1
        V[2*p + k0, col] = 1
        V[1*p + k0, col] = -1
        W[k0*n + 0, col] = 1
        W[k0*n + 1, col] = 1
        col += 1
        
        # M5 = (A[0,1] + A[0,2]) * B[2,k1]
        U[0*m + 1, col] = 1
        U[0*m + 2, col] = 1
        V[2*p + k1, col] = 1
        W[k0*n + 0, col] = -1
        W[k1*n + 0, col] = 1
        col += 1
        
        # M6 = (A[1,1] - A[0,1]) * (B[1,k0] + B[1,k1])
        U[1*m + 1, col] = 1
        U[0*m + 1, col] = -1
        V[1*p + k0, col] = 1
        V[1*p + k1, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        
        # M7 = (A[0,2] - A[1,2]) * (B[2,k0] + B[2,k1])
        U[0*m + 2, col] = 1
        U[1*m + 2, col] = -1
        V[2*p + k0, col] = 1
        V[2*p + k1, col] = 1
        W[k0*n + 0, col] = 1
        col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
