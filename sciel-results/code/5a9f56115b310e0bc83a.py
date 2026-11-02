import numpy as np
import time

def construct():
    """
    Returns a rank-minimal tensor decomposition for 2x2 * 2x14 matrix multiplication.
    The rank is 49, which is provably minimal.
    """
    n, m, p = 2, 2, 14
    size_a = n * m  # 4
    size_b = m * p  # 28
    size_c = p * n  # 28
    
    rank = 49
    U = np.zeros((size_a, rank), dtype=np.float64)
    V = np.zeros((size_b, rank), dtype=np.float64)
    W = np.zeros((size_c, rank), dtype=np.float64)
    
    # A indices:
    # 0: A00
    # 1: A01
    # 2: A10
    # 3: A11
    
    for l in range(7):
        # B indices for block l (columns 2l, 2l+1)
        # B is 2x14. Index = row * 14 + col.
        b_00 = 0 * p + 2 * l       # B[0, 2l]
        b_01 = 0 * p + (2 * l + 1) # B[0, 2l+1]
        b_10 = 1 * p + 2 * l       # B[1, 2l]
        b_11 = 1 * p + (2 * l + 1) # B[1, 2l+1]
        
        # W indices for block l (rows 2l, 2l+1)
        # W is 14x2. Index = row * 2 + col.
        w_00 = 2 * l * n + 0       # C[2l, 0]
        w_01 = 2 * l * n + 1       # C[2l, 1]
        w_10 = (2 * l + 1) * n + 0 # C[2l+1, 0]
        w_11 = (2 * l + 1) * n + 1 # C[2l+1, 1]
        
        base_col = 7 * l
        
        # Term 1: P1 = (A00 + A11) * (B00 + B11)
        # U: A00=1, A11=1
        U[0, base_col] = 1
        U[3, base_col] = 1
        # V: B00=1, B11=1
        V[b_00, base_col] = 1
        V[b_11, base_col] = 1
        # W: C00=1, C11=1
        W[w_00, base_col] = 1
        W[w_11, base_col] = 1
        
        # Term 2: P2 = (A10 + A11) * B00
        # U: A10=1, A11=1
        U[2, base_col + 1] = 1
        U[3, base_col + 1] = 1
        # V: B00=1
        V[b_00, base_col + 1] = 1
        # W: C10=1, C11=-1
        W[w_10, base_col + 1] = 1
        W[w_11, base_col + 1] = -1
        
        # Term 3: P3 = A00 * (B01 - B11)
        # U: A00=1
        U[0, base_col + 2] = 1
        # V: B01=1, B11=-1
        V[b_01, base_col + 2] = 1
        V[b_11, base_col + 2] = -1
        # W: C01=1, C11=1
        W[w_01, base_col + 2] = 1
        W[w_11, base_col + 2] = 1
        
        # Term 4: P4 = A11 * (B10 - B00)
        # U: A11=1
        U[3, base_col + 3] = 1
        # V: B10=1, B00=-1
        V[b_10, base_col + 3] = 1
        V[b_00, base_col + 3] = -1
        # W: C00=1, C10=1
        W[w_00, base_col + 3] = 1
        W[w_10, base_col + 3] = 1
        
        # Term 5: P5 = (A00 + A01) * B11
        # U: A00=1, A01=1
        U[0, base_col + 4] = 1
        U[1, base_col + 4] = 1
        # V: B11=1
        V[b_11, base_col + 4] = 1
        # W: C00=-1, C01=1
        W[w_00, base_col + 4] = -1
        W[w_01, base_col + 4] = 1
        
        # Term 6: P6 = (A10 - A00) * (B00 + B01)
        # U: A10=1, A00=-1
        U[2, base_col + 5] = 1
        U[0, base_col + 5] = -1
        # V: B00=1, B01=1
        V[b_00, base_col + 5] = 1
        V[b_01, base_col + 5] = 1
        # W: C11=1
        W[w_11, base_col + 5] = 1
        
        # Term 7: P7 = (A01 - A11) * (B10 + B11)
        # U: A01=1, A11=-1
        U[1, base_col + 6] = 1
        U[3, base_col + 6] = -1
        # V: B10=1, B11=1
        V[b_10, base_col + 6] = 1
        V[b_11, base_col + 6] = 1
        # W: C00=1
        W[w_00, base_col + 6] = 1

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
