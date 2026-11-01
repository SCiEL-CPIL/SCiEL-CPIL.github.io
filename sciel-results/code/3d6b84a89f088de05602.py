"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Strassen's rank-7 decomposition for 2x2 matrix multiplication.
    
    The matrix multiplication tensor T[i,j,k] = e_{i,j} e_{j,k} e_{k,i}
    (with W indexed as W[k*n+i, r] per the trace convention) can be
    decomposed into 7 rank-1 terms.
    
    U has shape (n*m, rank) = (4, 7)
    V has shape (m*p, rank) = (4, 7)
    W has shape (p*n, rank) = (4, 7)
    
    Indexing:
      U row: i*m + j  (i in 0..n-1, j in 0..m-1)
      V row: j*p + k  (j in 0..m-1, k in 0..p-1)
      W row: k*n + i  (k in 0..p-1, i in 0..n-1)  [trace convention]
    
    Strassen's 7 products:
      M1 = (A00+A11)*(B00+B11)
      M2 = (A10+A11)*B00
      M3 = A00*(B01-B11)
      M4 = A11*(B10-B00)
      M5 = (A00+A01)*B11
      M6 = (A10-A00)*(B00+B01)
      M7 = (A01-A11)*(B10+B11)
    
    Result:
      C00 = M1 + M4 - M5 + M7
      C01 = M3 + M5
      C10 = M2 + M4
      C11 = M1 - M2 + M3 + M6
    """
    n, m, p = DIMENSIONS
    rank = 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # Each column c corresponds to one of the 7 products M_c
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    # M1 = (A00+A11)*(B00+B11)
    # U: A00 -> row 0*m+0=0, A11 -> row 1*m+1=3
    # V: B00 -> row 0*p+0=0, B11 -> row 1*p+1=3
    # W: C00 -> row 0*n+0=0, C11 -> row 1*n+1=3
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)*B00
    # U: A10 -> row 1*m+0=2, A11 -> row 3
    # V: B00 -> row 0
    # W: C10 -> row 1*n+0=2, C11 -> row 3 (with -1 for C11)
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A00*(B01-B11)
    # U: A00 -> row 0
    # V: B01 -> row 0*p+1=1, B11 -> row 3 (with -1)
    # W: C00 -> row 0 (with -1), C01 -> row 0*n+1=1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[0][2] = -1; W[1][2] = 1
    
    # M4 = A11*(B10-B00)
    # U: A11 -> row 3
    # V: B10 -> row 1*p+0=2, B00 -> row 0 (with -1)
    # W: C00 -> row 0, C10 -> row 2
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)*B11
    # U: A00 -> row 0, A01 -> row 0*m+1=1
    # V: B11 -> row 3
    # W: C00 -> row 0 (with -1), C01 -> row 1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A10-A00)*(B00+B01)
    # U: A10 -> row 2, A00 -> row 0 (with -1)
    # V: B00 -> row 0, B01 -> row 1
    # W: C11 -> row 3
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)*(B10+B11)
    # U: A01 -> row 1, A11 -> row 3 (with -1)
    # V: B10 -> row 2, B11 -> row 3
    # W: C00 -> row 0
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
