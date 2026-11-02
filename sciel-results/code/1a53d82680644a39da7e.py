import numpy as np

def construct():
    """
    Returns the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    
    The tensor rank of the 2x2 matrix multiplication tensor is 7.
    The decomposition satisfies:
      sum_{r=0}^{6} U[i*m+a, r] * V[a*p+k, r] * W[k*n+i, r] = 1
      for all i in {0,1}, a in {0,1}, k in {0,1}.
      
    Indexing:
      U: (4, 7) -> U[i*2 + a, r] corresponds to A[i, a]
      V: (4, 7) -> V[a*2 + k, r] corresponds to B[a, k]
      W: (4, 7) -> W[k*2 + i, r] corresponds to C[i, k] coefficient
    """
    n, m, p = 2, 2, 2
    rank = 7
    
    # Initialize with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Indices:
    # A00: U[0, :], A01: U[1, :], A10: U[2, :], A11: U[3, :]
    # B00: V[0, :], B01: V[1, :], B10: V[2, :], B11: V[3, :]
    # C00: W[0, :], C10: W[1, :], C01: W[2, :], C11: W[3, :]
    # Note: W index is k*n + i. 
    # C00 (i=0,k=0) -> 0*2+0=0
    # C10 (i=1,k=0) -> 0*2+1=1
    # C01 (i=0,k=1) -> 1*2+0=2
    # C11 (i=1,k=1) -> 1*2+1=3
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0][0] = 1  # A00
    U[3][0] = 1  # A11
    V[0][0] = 1  # B00
    V[3][0] = 1  # B11
    
    # M2 = (A10 + A11)B00
    U[2][1] = 1  # A10
    U[3][1] = 1  # A11
    V[0][1] = 1  # B00
    
    # M3 = A00(B01 - B11)
    U[0][2] = 1  # A00
    V[1][2] = 1  # B01
    V[3][2] = -1 # -B11
    
    # M4 = A11(B10 - B00)
    U[3][3] = 1  # A11
    V[2][3] = 1  # B10
    V[0][3] = -1 # -B00
    
    # M5 = (A00 + A01)B11
    U[0][4] = 1  # A00
    U[1][4] = 1  # A01
    V[3][4] = 1  # B11
    
    # M6 = (A10 - A00)(B00 + B01)
    U[2][5] = 1  # A10
    U[0][5] = -1 # -A00
    V[0][5] = 1  # B00
    V[1][5] = 1  # B01
    
    # M7 = (A01 - A11)(B10 + B11)
    U[1][6] = 1  # A01
    U[3][6] = -1 # -A11
    V[2][6] = 1  # B10
    V[3][6] = 1  # B11
    
    # Combine to get C
    # C00 = M1 + M4 - M5 + M7
    W[0][0] = 1  # M1
    W[0][3] = 1  # M4
    W[0][4] = -1 # -M5
    W[0][6] = 1  # M7
    
    # C10 = M2 + M4
    W[1][1] = 1  # M2
    W[1][3] = 1  # M4
    
    # C01 = M3 + M5
    W[2][2] = 1  # M3
    W[2][4] = 1  # M5
    
    # C11 = M1 - M2 + M3 + M6
    W[3][0] = 1  # M1
    W[3][1] = -1 # -M2
    W[3][2] = 1  # M3
    W[3][5] = 1  # M6
    
    return {"U": U, "V": V, "W": W}
