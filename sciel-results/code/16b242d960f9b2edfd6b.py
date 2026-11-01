"""
Strassen rank-7 decomposition for 2x2 matrix multiplication.
This is the rank-minimal decomposition.
"""
import numpy as np

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Return Strassen's rank-7 tensor decomposition for 2x2 matrix multiplication.
    
    Returns:
    dict: {'U': list, 'V': list, 'W': list}
    U: shape (4, 7), V: shape (4, 7), W: shape (4, 7)
    Entries are integers.
    """
    n, m, p = DIMENSIONS
    rank = 7
    
    # Initialize U, V, W with zeros
    # U[i*m + j, c]
    # V[j*p + k, c]
    # W[k*n + i, c]
    
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    
    # Strassen's 7 factors
    # Indices:
    # A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C: 00, 01, 10, 11 -> 0, 1, 2, 3 (but W index is k*n+i)
    # C[0,0] -> k=0, i=0 -> idx 0
    # C[0,1] -> k=1, i=0 -> idx 2
    # C[1,0] -> k=0, i=1 -> idx 1
    # C[1,1] -> k=1, i=1 -> idx 3
    
    # M1 = (A00+A11)(B00+B11)
    # U: A00(0), A11(3) -> [1, 0, 0, 1]
    # V: B00(0), B11(3) -> [1, 0, 0, 1]
    # W: C00(0), C11(3) -> [1, 0, 0, 1] (C00 idx 0, C11 idx 3)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    # U: A10(2), A11(3) -> [0, 0, 1, 1]
    # V: B00(0) -> [1, 0, 0, 0]
    # W: C10(1), C11(3) -> [0, 1, 0, -1] (C10 idx 1, C11 idx 3)
    # C10 = M2 + M4. C11 = M1 + M3 - M2 + M6.
    # So M2 contributes +1 to C10 and -1 to C11.
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B01-B11)
    # U: A00(0) -> [1, 0, 0, 0]
    # V: B01(1), B11(3) -> [0, 1, 0, -1]
    # W: C01(2), C11(3) -> [0, 0, 1, 1] (C01 idx 2, C11 idx 3)
    # C01 = M3 + M5. C11 = M1 + M3 - M2 + M6.
    # So M3 contributes +1 to C01 and +1 to C11.
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10-B00)
    # U: A11(3) -> [0, 0, 0, 1]
    # V: B10(2), B00(0) -> [0, 0, 1, 0] with sign?
    # B10 - B00 -> V[2]=1, V[0]=-1
    # W: C00(0), C10(1) -> [1, 1, 0, 0] (C00 idx 0, C10 idx 1)
    # C00 = M1 - M5 + M4 + M7. C10 = M2 + M4.
    # So M4 contributes +1 to C00 and +1 to C10.
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A00+A01)B11
    # U: A00(0), A01(1) -> [1, 1, 0, 0]
    # V: B11(3) -> [0, 0, 0, 1]
    # W: C00(0), C01(2) -> [-1, 0, 1, 0] (C00 idx 0, C01 idx 2)
    # C00 = ... - M5 ... C01 = M3 + M5.
    # So M5 contributes -1 to C00 and +1 to C01.
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    # U: A10(2), A00(0) -> [0, 0, 1, 0] with sign?
    # A10 - A00 -> U[2]=1, U[0]=-1
    # V: B00(0), B01(1) -> [1, 1, 0, 0]
    # W: C11(3) -> [0, 0, 0, 1]
    # C11 = ... + M6
    # So M6 contributes +1 to C11.
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    # U: A01(1), A11(3) -> [0, 1, 0, -1]
    # V: B10(2), B11(3) -> [0, 0, 1, 1]
    # W: C00(0) -> [1, 0, 0, 0]
    # C00 = ... + M7
    # So M7 contributes +1 to C00.
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
