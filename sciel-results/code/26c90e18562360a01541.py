import numpy as np
import random
from fractions import Fraction

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    # Strassen decomposition for 2x2*2x2 (rank 7) + schoolbook for 2x2*2x1 (rank 4) = rank 11
    n, m, p = 2, 2, 3
    rank = 11

    # U is (n*m, rank) = (4, 11)
    # V is (m*p, rank) = (6, 11)
    # W is (p*n, rank) = (6, 11)
    
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(6)]
    W = [[0]*rank for _ in range(6)]
    
    # Indices:
    # U row: i*m + j  (i in 0..1, j in 0..1) -> A_{ij}
    # V row: j*p + k  (j in 0..1, k in 0..2) -> B_{jk}
    # W row: k*n + i  (k in 0..2, i in 0..1) -> C_{ki} (output index is k*n+i per trace convention W[k*n+i, r])
    
    # Strassen terms for first 2 columns of B (k=0,1), producing first 2 columns of C (k=0,1)
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B11)
    # M4 = A22(B22-B12)  -> Wait, standard Strassen uses B indices. Let's use 0-indexed.
    # A is 2x2, B is 2x2 (for the 2x2*2x2 part)
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    
    # Strassen (standard):
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B00)
    # M4 = A11 * (B11 - B01)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 + M3 - M2 + M4
    
    # Let's assign columns 0..6 to M1..M7
    col = 0
    # M1
    U[0][col] = 1  # A00
    U[3][col] = 1  # A11 (index 1*2+1 = 3)
    V[0][col] = 1  # B00 (index 0*3+0 = 0)
    V[3][col] = 1  # B11 (index 1*3+1 = 4) -> Wait, B is 2x3. For the 2x2 part, B columns 0,1.
                   # V row for B_{jk}: j*p+k. 
                   # B00 -> j=0,k=0 -> 0*3+0=0
                   # B11 -> j=1,k=1 -> 1*3+1=4
                   # B01 -> j=0,k=1 -> 0*3+1=1
                   # B10 -> j=1,k=0 -> 1*3+0=3
    col += 1
    
    # M2
    U[2][col] = 1  # A10 (index 1*2+0 = 2)
    U[3][col] = 1  # A11
    V[0][col] = 1  # B00
    col += 1
    
    # M3
    U[0][col] = 1  # A00
    V[3][col] = 1  # B10
    V[0][col] = -1 # B00
    col += 1
    
    # M4
    U[3][col] = 1  # A11
    V[4][col] = 1  # B11
    V[1][col] = -1 # B01
    col += 1
    
    # M5
    U[0][col] = 1  # A00
    U[1][col] = 1  # A01 (index 0*2+1 = 1)
    V[4][col] = 1  # B11
    col += 1
    
    # M6
    U[2][col] = 1  # A10
    U[0][col] = -1 # A00
    V[0][col] = 1  # B00
    V[1][col] = 1  # B01
    col += 1
    
    # M7
    U[1][col] = 1  # A01
    U[3][col] = -1 # A11
    V[3][col] = 1  # B10
    V[4][col] = 1  # B11
    col += 1
    
    # Now fill W for Strassen part
    # C00 (k=0,i=0 -> W index 0*2+0=0) = M1 + M4 - M5 + M7
    # C01 (k=1,i=0 -> W index 1*2+0=2) = M3 + M5
    # C10 (k=0,i=1 -> W index 0*2+1=1) = M2 + M6
    # C11 (k=1,i=1 -> W index 1*2+1=3) = M1 + M3 - M2 + M4
    
    # W row 0 (C00): M1, M4, -M5, M7
    W[0][0] = 1
    W[0][3] = 1
    W[0][4] = -1
    W[0][6] = 1
    
    # W row 1 (C10): M2, M6
    W[1][1] = 1
    W[1][5] = 1
    
    # W row 2 (C01): M3, M5
    W[2][2] = 1
    W[2][4] = 1
    
    # W row 3 (C11): M1, M3, -M2, M4
    W[3][0] = 1
    W[3][2] = 1
    W[3][1] = -1
    W[3][3] = 1
    
    # Schoolbook for the 3rd column of B (k=2), producing 3rd column of C (k=2)
    # C_{i,2} = A_{i,0}B_{0,2} + A_{i,1}B_{1,2}
    # 4 terms: (i=0,j=0), (i=0,j=1), (i=1,j=0), (i=1,j=1)
    
    # Term 8: A00 * B02
    col = 7
    U[0][col] = 1  # A00
    V[2][col] = 1  # B02 (index 0*3+2 = 2)
    W[4][col] = 1  # C02 (index 2*2+0 = 4)
    col += 1
    
    # Term 9: A01 * B12
    U[1][col] = 1  # A01
    V[5][col] = 1  # B12 (index 1*3+2 = 5)
    W[4][col] = 1  # C02
    col += 1
    
    # Term 10: A10 * B02
    U[2][col] = 1  # A10
    V[2][col] = 1  # B02
    W[5][col] = 1  # C12 (index 2*2+1 = 5)
    col += 1
    
    # Term 11: A11 * B12
    U[3][col] = 1  # A11
    V[5][col] = 1  # B12
    W[5][col] = 1  # C12
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
