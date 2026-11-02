import numpy as np

def construct():
    """
    Returns the rank-minimal exact tensor decomposition for 2x2 matrix multiplication.
    The minimum rank is 7 (Strassen's algorithm).
    
    Interface:
    U: shape (n*m, rank) = (4, 7)
    V: shape (m*p, rank) = (4, 7)
    W: shape (p*n, rank) = (4, 7)
    
    Indexing:
    U[i*m + a, r]
    V[a*p + k, r]
    W[k*n + i, r]
    """
    n, m, p = 2, 2, 2
    rank = 7
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen's 7 products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Indices for U (A matrix):
    # A00: i=0, a=0 -> idx 0
    # A01: i=0, a=1 -> idx 1
    # A10: i=1, a=0 -> idx 2
    # A11: i=1, a=1 -> idx 3
    
    # Indices for V (B matrix):
    # B00: a=0, k=0 -> idx 0
    # B01: a=0, k=1 -> idx 1
    # B10: a=1, k=0 -> idx 2
    # B11: a=1, k=1 -> idx 3
    
    # Indices for W (C matrix output):
    # C00: k=0, i=0 -> idx 0
    # C01: k=1, i=0 -> idx 2
    # C10: k=0, i=1 -> idx 1
    # C11: k=1, i=1 -> idx 3
    
    # M1
    U[0, 0] = 1  # A00
    U[3, 0] = 1  # A11
    V[0, 0] = 1  # B00
    V[3, 0] = 1  # B11
    
    # M2
    U[2, 1] = 1  # A10
    U[3, 1] = 1  # A11
    V[0, 1] = 1  # B00
    
    # M3
    U[0, 2] = 1  # A00
    V[1, 2] = 1  # B01
    V[3, 2] = -1 # -B11
    
    # M4
    U[3, 3] = 1  # A11
    V[2, 3] = 1  # B10
    V[0, 3] = -1 # -B00
    
    # M5
    U[0, 4] = 1  # A00
    U[1, 4] = 1  # A01
    V[3, 4] = 1  # B11
    
    # M6
    U[2, 5] = 1  # A10
    U[0, 5] = -1 # -A00
    V[0, 5] = 1  # B00
    V[1, 5] = 1  # B01
    
    # M7
    U[1, 6] = 1  # A01
    U[3, 6] = -1 # -A11
    V[2, 6] = 1  # B10
    V[3, 6] = 1  # B11
    
    # Reconstruction of C:
    # C00 = M1 + M4 - M5 + M7
    W[0, 0] = 1   # M1
    W[0, 3] = 1   # M4
    W[0, 4] = -1  # -M5
    W[0, 6] = 1   # M7
    
    # C01 = M3 + M5
    W[2, 2] = 1   # M3
    W[2, 4] = 1   # M5
    
    # C10 = M2 + M4
    W[1, 1] = 1   # M2
    W[1, 3] = 1   # M4
    
    # C11 = M1 - M2 + M3 + M6
    W[3, 0] = 1   # M1
    W[3, 1] = -1  # -M2
    W[3, 2] = 1   # M3
    W[3, 5] = 1   # M6
    
    return {"U": U, "V": V, "W": W}
