"""
Rank-minimal exact tensor decomposition for 2x2 matrix multiplication.
Uses Strassen's algorithm, which achieves the theoretical minimum rank of 7.
"""
import numpy as np

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-7 exact tensor decomposition for 2x2 matrix multiplication.
    
    Interface:
    U: shape (n*m, rank) = (4, 7). Row index i*m + j corresponds to A[i,j].
    V: shape (m*p, rank) = (4, 7). Row index j*p + k corresponds to B[j,k].
    W: shape (p*n, rank) = (4, 7). Row index k*n + i corresponds to C[i,k] (trace convention).
    
    The decomposition satisfies:
    T[i,j,k] = sum_{r=0}^{6} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    where T[i,j,k] = 1 for the multiplication tensor (since C[i,k] = sum_j A[i,j]B[j,k]).
    """
    n, m, p = 2, 2, 2
    rank = 7
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen's 7 products:
    # M0 = (A00 + A11)(B00 + B11)
    # M1 = (A10 + A11)B00
    # M2 = A00(B01 - B11)
    # M3 = A11(B10 - B00)
    # M4 = (A00 + A01)B11
    # M5 = (A10 - A00)(B00 + B01)
    # M6 = (A01 - A11)(B10 + B11)
    
    # Reconstruction of C:
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # Indices:
    # A: (i,j) -> i*2 + j. A00=0, A01=1, A10=2, A11=3
    # B: (j,k) -> j*2 + k. B00=0, B01=1, B10=2, B11=3
    # C: (i,k) -> k*2 + i (trace convention W[k*n+i]).
    # C00 (i=0,k=0) -> 0
    # C01 (i=0,k=1) -> 2
    # C10 (i=1,k=0) -> 1
    # C11 (i=1,k=1) -> 3
    
    # M0
    U[0, 0] = 1  # A00
    U[3, 0] = 1  # A11
    V[0, 0] = 1  # B00
    V[3, 0] = 1  # B11
    W[0, 0] = 1  # C00
    W[3, 0] = 1  # C11
    
    # M1
    U[2, 1] = 1  # A10
    U[3, 1] = 1  # A11
    V[0, 1] = 1  # B00
    W[1, 1] = 1  # C10
    W[3, 1] = -1 # C11
    
    # M2
    U[0, 2] = 1  # A00
    V[1, 2] = 1  # B01
    V[3, 2] = -1 # B11
    W[2, 2] = 1  # C01
    W[3, 2] = 1  # C11
    
    # M3
    U[3, 3] = 1  # A11
    V[2, 3] = 1  # B10
    V[0, 3] = -1 # B00
    W[0, 3] = 1  # C00
    W[1, 3] = 1  # C10
    
    # M4
    U[0, 4] = 1  # A00
    U[1, 4] = 1  # A01
    V[3, 4] = 1  # B11
    W[0, 4] = -1 # C00
    W[2, 4] = 1  # C01
    
    # M5
    U[2, 5] = 1  # A10
    U[0, 5] = -1 # A00
    V[0, 5] = 1  # B00
    V[1, 5] = 1  # B01
    W[3, 5] = 1  # C11
    
    # M6
    U[1, 6] = 1  # A01
    U[3, 6] = -1 # A11
    V[2, 6] = 1  # B10
    V[3, 6] = 1  # B11
    W[0, 6] = 1  # C00
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
