import numpy as np
import time

def construct():
    """
    Returns a rank-7 exact decomposition of the 2x2x2 matrix multiplication tensor.
    
    This corresponds to Strassen's algorithm for 2x2 matrix multiplication.
    The decomposition satisfies:
        T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    where T[i,j,k] = 1 for all i, j, k in {0, 1}.
    
    Returns:
        dict with keys 'U', 'V', 'W'
        U: shape (4, 7) list of lists
        V: shape (4, 7) list of lists
        W: shape (4, 7) list of lists
    """
    # Strassen's 7 products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Output reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Indexing conventions:
    # U index: i*m + j  (0: A00, 1: A01, 2: A10, 3: A11)
    # V index: j*p + k  (0: B00, 1: B01, 2: B10, 3: B11)
    # W index: k*n + i  (0: C00, 1: C10, 2: C01, 3: C11)
    
    # Define the factor matrices for Strassen's algorithm
    # Each column r corresponds to one of the 7 products M1..M7
    
    # U: (4, 7)
    # Rows: A00, A01, A10, A11
    U = [
        [1.0,  0.0,  1.0,  0.0,  1.0, -1.0,  0.0], # A00
        [0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  1.0], # A01
        [0.0,  1.0,  0.0,  0.0,  0.0,  1.0,  0.0], # A10
        [1.0,  1.0,  0.0,  1.0,  0.0,  0.0, -1.0]  # A11
    ]
    
    # V: (4, 7)
    # Rows: B00, B01, B10, B11
    V = [
        [1.0,  1.0,  0.0, -1.0,  0.0,  1.0,  0.0], # B00
        [0.0,  0.0,  1.0,  0.0,  0.0,  1.0,  0.0], # B01
        [0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  1.0], # B10
        [1.0,  0.0, -1.0,  0.0,  1.0,  0.0,  1.0]  # B11
    ]
    
    # W: (4, 7)
    # W index: k*n + i
    # k=0, i=0 -> idx 0 (C[0,0])
    # k=0, i=1 -> idx 1 (C[1,0])
    # k=1, i=0 -> idx 2 (C[0,1])
    # k=1, i=1 -> idx 3 (C[1,1])
    
    # Reconstruction:
    # C[0,0] = M1 + M4 - M5 + M7
    # C[1,0] = M2 + M4
    # C[0,1] = M3 + M5
    # C[1,1] = M1 - M2 + M3 + M6
    
    W = [
        [1.0,  0.0,  0.0,  1.0, -1.0,  0.0,  1.0], # C00 (idx 0)
        [0.0,  1.0,  0.0,  1.0,  0.0,  0.0,  0.0], # C10 (idx 1)
        [0.0,  0.0,  1.0,  0.0,  1.0,  0.0,  0.0], # C01 (idx 2)
        [1.0, -1.0,  1.0,  0.0,  0.0,  1.0,  0.0]  # C11 (idx 3)
    ]
    
    return {
        "U": U,
        "V": V,
        "W": W
    }
