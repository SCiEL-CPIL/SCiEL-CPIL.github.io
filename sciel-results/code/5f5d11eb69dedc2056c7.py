"""
Program to find the rank-minimal exact tensor decomposition for 2x2 matrix multiplication.
The minimum rank for 2x2 matrix multiplication is 7 (Strassen's algorithm).
"""

def construct():
    """
    Returns a rank-7 exact tensor decomposition for 2x2 matrix multiplication.
    
    Interface:
    U: shape (n*m, rank) = (4, 7)
    V: shape (m*p, rank) = (4, 7)
    W: shape (p*n, rank) = (4, 7)
    
    Indexing conventions:
    U[i*m + a, r]
    V[a*p + k, r]
    W[k*n + i, r]
    
    Decomposition identity:
    C[i, k] = sum_r W[k*n + i, r] * (sum_{i', a'} U[i'*m + a', r] * A[i', a']) * (sum_{a', k'} V[a'*p + k', r] * B[a', k'])
    """
    n, m, p = 2, 2, 2
    rank = 7
    
    # Initialize matrices
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen's 7 products
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # r=0: M1
    # U part: A00 + A11
    U[0*2+0][0] = 1.0  # A00
    U[1*2+1][0] = 1.0  # A11
    # V part: B00 + B11
    V[0*2+0][0] = 1.0  # B00
    V[1*2+1][0] = 1.0  # B11
    # W part: C00 += M1, C11 += M1
    W[0*2+0][0] = 1.0  # C[0,0] (k=0, i=0)
    W[1*2+1][0] = 1.0  # C[1,1] (k=1, i=1)
    
    # r=1: M2
    # U part: A10 + A11
    U[1*2+0][1] = 1.0  # A10
    U[1*2+1][1] = 1.0  # A11
    # V part: B00
    V[0*2+0][1] = 1.0  # B00
    # W part: C10 += M2
    W[0*2+1][1] = 1.0  # C[1,0] (k=0, i=1)
    
    # r=2: M3
    # U part: A00
    U[0*2+0][2] = 1.0  # A00
    # V part: B01 - B11
    V[0*2+1][2] = 1.0  # B01
    V[1*2+1][2] = -1.0 # -B11
    # W part: C01 += M3
    W[1*2+0][2] = 1.0  # C[0,1] (k=1, i=0)
    
    # r=3: M4
    # U part: A11
    U[1*2+1][3] = 1.0  # A11
    # V part: B10 - B00
    V[1*2+0][3] = 1.0  # B10
    V[0*2+0][3] = -1.0 # -B00
    # W part: C00 += M4, C10 += M4
    W[0*2+0][3] = 1.0  # C[0,0] (k=0, i=0)
    W[0*2+1][3] = 1.0  # C[1,0] (k=0, i=1)
    
    # r=4: M5
    # U part: A00 + A01
    U[0*2+0][4] = 1.0  # A00
    U[0*2+1][4] = 1.0  # A01
    # V part: B11
    V[1*2+1][4] = 1.0  # B11
    # W part: C00 -= M5, C01 += M5
    W[0*2+0][4] = -1.0 # C[0,0] (k=0, i=0)
    W[1*2+0][4] = 1.0  # C[0,1] (k=1, i=0)
    
    # r=5: M6
    # U part: A10 - A00
    U[1*2+0][5] = 1.0  # A10
    U[0*2+0][5] = -1.0 # -A00
    # V part: B00 + B01
    V[0*2+0][5] = 1.0  # B00
    V[0*2+1][5] = 1.0  # B01
    # W part: C11 += M6
    W[1*2+1][5] = 1.0  # C[1,1] (k=1, i=1)
    
    # r=6: M7
    # U part: A01 - A11
    U[0*2+1][6] = 1.0  # A01
    U[1*2+1][6] = -1.0 # -A11
    # V part: B10 + B11
    V[1*2+0][6] = 1.0  # B10
    V[1*2+1][6] = 1.0  # B11
    # W part: C00 += M7
    W[0*2+0][6] = 1.0  # C[0,0] (k=0, i=0)
    
    return {"U": U, "V": V, "W": W}
