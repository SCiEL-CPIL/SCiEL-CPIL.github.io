"""
Program for 2x2 matrix multiplication rank minimization.
Implements the known optimal Strassen rank-7 decomposition directly.
The 2x2 matrix multiplication tensor has rank 7, so this is optimal.
"""
import numpy as np

DIMENSIONS = [2, 2, 2]

def construct():
    """
    Return the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    
    The decomposition satisfies:
      (A*B)[i, k] = sum_{a, r} U[i*m + a, r] * V[a*p + k, r] * W[k*n + i, r]
    
    where:
      U: shape (n*m, rank) = (4, 7)
      V: shape (m*p, rank) = (4, 7)
      W: shape (p*n, rank) = (4, 7)
    
    Indexing conventions:
      U[i*2 + a, r]  : coefficient of A[i,a] in r-th product
      V[a*2 + k, r]  : coefficient of B[a,k] in r-th product
      W[k*2 + i, r]  : coefficient for r-th product contributing to C[i,k]
    
    Strassen's 7 products for 2x2 matrices:
      M1 = (A00 + A11)(B00 + B11)
      M2 = (A10 + A11)(B00)
      M3 = (A00)(B01 - B11)
      M4 = (A11)(B10 - B00)
      M5 = (A00 + A01)(B11)
      M6 = (A10 - A00)(B00 + B01)
      M7 = (A01 - A11)(B10 + B11)
    
    Output:
      C00 = M1 + M4 - M5 + M7
      C01 = M3 + M5
      C10 = M2 + M4
      C11 = M1 - M2 + M3 + M6
    """
    n, m, p = 2, 2, 2
    rank = 7
    
    # Initialize with zeros
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # M1: (A00 + A11)(B00 + B11)
    U[0*2+0][0] = 1   # A00
    U[1*2+1][0] = 1   # A11
    V[0*2+0][0] = 1   # B00
    V[1*2+1][0] = 1   # B11
    
    # M2: (A10 + A11)(B00)
    U[1*2+0][1] = 1   # A10
    U[1*2+1][1] = 1   # A11
    V[0*2+0][1] = 1   # B00
    
    # M3: (A00)(B01 - B11)
    U[0*2+0][2] = 1   # A00
    V[0*2+1][2] = 1   # B01
    V[1*2+1][2] = -1  # -B11
    
    # M4: (A11)(B10 - B00)
    U[1*2+1][3] = 1   # A11
    V[1*2+0][3] = 1   # B10
    V[0*2+0][3] = -1  # -B00
    
    # M5: (A00 + A01)(B11)
    U[0*2+0][4] = 1   # A00
    U[0*2+1][4] = 1   # A01
    V[1*2+1][4] = 1   # B11
    
    # M6: (A10 - A00)(B00 + B01)
    U[1*2+0][5] = 1   # A10
    U[0*2+0][5] = -1  # -A00
    V[0*2+0][5] = 1   # B00
    V[0*2+1][5] = 1   # B01
    
    # M7: (A01 - A11)(B10 + B11)
    U[0*2+1][6] = 1   # A01
    U[1*2+1][6] = -1  # -A11
    V[1*2+0][6] = 1   # B10
    V[1*2+1][6] = 1   # B11
    
    # W[k*n + i, r]: coefficient of r-th product in C[i,k]
    # C00 = M1 + M4 - M5 + M7  -> k=0, i=0 -> W[0*2+0] = W[0]
    W[0][0] = 1   # M1
    W[0][3] = 1   # M4
    W[0][4] = -1  # -M5
    W[0][6] = 1   # M7
    
    # C01 = M3 + M5  -> k=1, i=0 -> W[1*2+0] = W[2]
    W[2][2] = 1   # M3
    W[2][4] = 1   # M5
    
    # C10 = M2 + M4  -> k=0, i=1 -> W[0*2+1] = W[1]
    W[1][1] = 1   # M2
    W[1][3] = 1   # M4
    
    # C11 = M1 - M2 + M3 + M6  -> k=1, i=1 -> W[1*2+1] = W[3]
    W[3][0] = 1   # M1
    W[3][1] = -1  # -M2
    W[3][2] = 1   # M3
    W[3][5] = 1   # M6
    
    # Verify the decomposition
    # For random matrices A and B, check that the decomposition gives correct C
    np.random.seed(42)
    for _ in range(10):
        A = np.random.randint(-5, 6, size=(n, m))
        B = np.random.randint(-5, 6, size=(m, p))
        C_expected = A @ B
        
        # Compute using decomposition:
        # C[i,k] = sum_{a,r} U[i*m+a, r] * V[a*p+k, r] * W[k*n+i, r]
        C_computed = np.zeros((n, p), dtype=int)
        for i in range(n):
            for k in range(p):
                s = 0
                for a in range(m):
                    for r in range(rank):
                        s += U[i*m + a, r] * V[a*p + k, r] * W[k*n + i, r]
                C_computed[i, k] = s
        
        if not np.array_equal(C_expected, C_computed):
            raise ValueError(f"Decomposition failed verification!\n"
                           f"A = {A}\nB = {B}\n"
                           f"Expected: {C_expected}\nGot: {C_computed}")
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
