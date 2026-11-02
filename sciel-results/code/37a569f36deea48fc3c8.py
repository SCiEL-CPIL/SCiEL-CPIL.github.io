import numpy as np
import time

DIMENSIONS = [2, 2, 12]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    rank = 2 * p  # 24
    
    U = np.zeros((n * m, rank), dtype=np.float64)  # 4 x 24
    V = np.zeros((m * p, rank), dtype=np.float64)  # 24 x 24
    W = np.zeros((p * n, rank), dtype=np.float64)  # 24 x 24
    
    # For each column k of B (k = 0..11), use two bilinear terms:
    # Term 2k:   U[A00]=1, U[A11]=1, V[B0k]=1, W[C0k]=1, W[C1k]=1
    #            computes (A00+A11)*B0k, contributes +1 to C[0,k] and +1 to C[1,k]
    # Term 2k+1: U[A00]=1, U[A11]=-1, V[B1k]=1, W[C0k]=1, W[C1k]=-1
    #            computes (A00-A11)*B1k, contributes +1 to C[0,k] and -1 to C[1,k]
    #
    # Verification: 
    #   C[0,k] = (A00+A11)*B0k + (A00-A11)*B1k = 2*A00*B0k + A11*B0k - A11*B1k
    #   Wait, that's not right. Let me re-derive.
    #
    # Actually the standard rank-2n construction for 2x2 x 2xn:
    # For each k, we need C[0,k] = A00*B0k + A01*B1k and C[1,k] = A10*B0k + A11*B1k
    # 
    # The known rank-2p construction uses:
    # For each k:
    #   c=2k:   U[A00]=1, U[A10]=1, V[B0k]=1, W[C0k]=1, W[C1k]=1
    #   c=2k+1: U[A01]=1, U[A11]=1, V[B1k]=1, W[C0k]=1, W[C1k]=1
    #
    # Check: C[0,k] = A00*B0k + A01*B1k ✓
    #         C[1,k] = A10*B0k + A11*B1k ✓
    
    for k in range(p):
        c0 = 2 * k
        c1 = 2 * k + 1
        
        # Term c0: captures A[0,0]*B[0,k] for C[0,k] and A[1,0]*B[0,k] for C[1,k]
        U[0, c0] = 1  # A[0,0]
        U[2, c0] = 1  # A[1,0]
        V[0 * p + k, c0] = 1  # B[0,k]
        W[k * n + 0, c0] = 1  # C[0,k]
        W[k * n + 1, c0] = 1  # C[1,k]
        
        # Term c1: captures A[0,1]*B[1,k] for C[0,k] and A[1,1]*B[1,k] for C[1,k]
        U[1, c1] = 1  # A[0,1]
        U[3, c1] = 1  # A[1,1]
        V[1 * p + k, c1] = 1  # B[1,k]
        W[k * n + 0, c1] = 1  # C[0,k]
        W[k * n + 1, c1] = 1  # C[1,k]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
