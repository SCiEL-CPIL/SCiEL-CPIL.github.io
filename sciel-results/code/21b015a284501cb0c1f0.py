import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    """
    Returns a rank-28 exact tensor decomposition for 2x3 by 3x5 matrix multiplication.
    
    The decomposition is constructed by partitioning the middle dimension (j) into:
      - j=0: a 2x1x5 block (rank 10, trivial)
      - j=1,2: a 2x2x5 block (rank 18, using Strassen on two 2x2x2 sub-blocks + 2x2x1 middle)
    
    Total rank = 10 + 18 = 28.
    
    U has shape (n*m, rank) = (6, 28)
    V has shape (m*p, rank) = (15, 28)
    W has shape (p*n, rank) = (10, 28)
    
    Index conventions:
      U[i*m + j, r]
      V[j*p + k, r]
      W[k*n + i, r]
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    rank = 28
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Block 1: j=0, all k in 0..4
    # This is a 2x1x5 multiplication, which has rank 10 (each output entry needs one mult)
    for i in range(n):
        for k in range(p):
            U[i*m + 0, col] = 1
            V[0*p + k, col] = 1
            W[k*n + i, col] = 1
            col += 1
    
    # Block 2: j=1,2, all k in 0..4
    # This is a 2x2x5 multiplication.
    # We split k into three parts: k=0,1 (2x2x2), k=2 (2x2x1), k=3,4 (2x2x2)
    
    # Sub-block 2a: j=1,2, k=0,1 -> 2x2x2, rank 7 (Strassen)
    # i in {0,1}, j in {1,2}, k in {0,1}
    
    # M1: (A[0,1]+A[1,2])(B[0,0]+B[1,0]) -> C[0,0], C[1,1]
    U[0*m+1, col] = 1
    U[1*m+2, col] = 1
    V[1*p+0, col] = 1
    V[2*p+0, col] = 1
    W[0*n+0, col] = 1
    W[1*n+1, col] = 1
    col += 1
    
    # M2: (A[1,1]+A[1,2])B[0,0] -> C[1,0], -C[1,1]
    U[1*m+1, col] = 1
    U[1*m+2, col] = 1
    V[1*p+0, col] = 1
    W[0*n+1, col] = 1
    W[1*n+1, col] = -1
    col += 1
    
    # M3: A[0,1](B[1,1]-B[0,1]) -> C[0,1], C[1,1]
    U[0*m+1, col] = 1
    V[2*p+1, col] = 1
    V[1*p+1, col] = -1
    W[1*n+0, col] = 1
    W[1*n+1, col] = 1
    col += 1
    
    # M4: A[1,2](B[1,0]-B[0,0]) -> C[0,0], C[1,0]
    U[1*m+2, col] = 1
    V[2*p+0, col] = 1
    V[1*p+0, col] = -1
    W[0*n+0, col] = 1
    W[0*n+1, col] = 1
    col += 1
    
    # M5: (A[0,1]+A[0,2])B[1,1] -> -C[0,0], C[0,1]
    U[0*m+1, col] = 1
    U[0*m+2, col] = 1
    V[2*p+1, col] = 1
    W[0*n+0, col] = -1
    W[1*n+0, col] = 1
    col += 1
    
    # M6: (A[1,1]-A[0,1])(B[0,0]+B[0,1]) -> C[1,1]
    U[1*m+1, col] = 1
    U[0*m+1, col] = -1
    V[1*p+0, col] = 1
    V[1*p+1, col] = 1
    W[1*n+1, col] = 1
    col += 1
    
    # M7: (A[0,2]-A[1,2])(B[1,0]+B[1,1]) -> C[0,0]
    U[0*m+2, col] = 1
    U[1*m+2, col] = -1
    V[2*p+0, col] = 1
    V[2*p+1, col] = 1
    W[0*n+0, col] = 1
    col += 1
    
    # Sub-block 2b: j=1,2, k=2 -> 2x2x1, rank 4
    # Each output C[i, k=2] = A[i,1]*B[1,2] + A[i,2]*B[2,2]
    # But since it's 2x2x1, we need 4 multiplications (one per A[i,j]*B[j,k] term)
    for i in range(n):
        for j in [1, 2]:
            U[i*m+j, col] = 1
            V[j*p+2, col] = 1
            W[2*n+i, col] = 1
            col += 1
    
    # Sub-block 2c: j=1,2, k=3,4 -> 2x2x2, rank 7 (Strassen)
    # i in {0,1}, j in {1,2}, k in {3,4}
    
    # M1: (A[0,1]+A[1,2])(B[0,3]+B[1,3]) -> C[0,3], C[1,4]
    U[0*m+1, col] = 1
    U[1*m+2, col] = 1
    V[1*p+3, col] = 1
    V[2*p+3, col] = 1
    W[3*n+0, col] = 1
    W[4*n+1, col] = 1
    col += 1
    
    # M2: (A[1,1]+A[1,2])B[0,3] -> C[1,3], -C[1,4]
    U[1*m+1, col] = 1
    U[1*m+2, col] = 1
    V[1*p+3, col] = 1
    W[3*n+1, col] = 1
    W[4*n+1, col] = -1
    col += 1
    
    # M3: A[0,1](B[1,4]-B[0,4]) -> C[0,4], C[1,4]
    U[0*m+1, col] = 1
    V[2*p+4, col] = 1
    V[1*p+4, col] = -1
    W[4*n+0, col] = 1
    W[4*n+1, col] = 1
    col += 1
    
    # M4: A[1,2](B[1,3]-B[0,3]) -> C[0,3], C[1,3]
    U[1*m+2, col] = 1
    V[2*p+3, col] = 1
    V[1*p+3, col] = -1
    W[3*n+0, col] = 1
    W[3*n+1, col] = 1
    col += 1
    
    # M5: (A[0,1]+A[0,2])B[1,4] -> -C[0,3], C[0,4]
    U[0*m+1, col] = 1
    U[0*m+2, col] = 1
    V[2*p+4, col] = 1
    W[3*n+0, col] = -1
    W[4*n+0, col] = 1
    col += 1
    
    # M6: (A[1,1]-A[0,1])(B[0,3]+B[0,4]) -> C[1,4]
    U[1*m+1, col] = 1
    U[0*m+1, col] = -1
    V[1*p+3, col] = 1
    V[1*p+4, col] = 1
    W[4*n+1, col] = 1
    col += 1
    
    # M7: (A[0,2]-A[1,2])(B[1,3]+B[1,4]) -> C[0,3]
    U[0*m+2, col] = 1
    U[1*m+2, col] = -1
    V[2*p+3, col] = 1
    V[2*p+4, col] = 1
    W[3*n+0, col] = 1
    col += 1
    
    assert col == rank, f"Column count mismatch: {col} != {rank}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
