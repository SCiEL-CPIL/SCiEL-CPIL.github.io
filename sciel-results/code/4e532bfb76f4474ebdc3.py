import numpy as np
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank 17 decomposition for 2x3x3 matrix multiplication
    # Constructed by combining:
    # 1. Strassen (rank 7) for 2x2x2 part
    # 2. Schoolbook (rank 4) for 2x1x2 part
    # 3. Schoolbook (rank 6) for 2x3x1 part
    
    # U shape: (n*m, 17) = (6, 17)
    # V shape: (m*p, 17) = (9, 17)
    # W shape: (p*n, 17) = (6, 17)
    
    # We define the non-zero entries directly.
    # U[r, col] = val
    # V[r, col] = val
    # W[r, col] = val
    
    U = np.zeros((6, 17))
    V = np.zeros((9, 17))
    W = np.zeros((6, 17))
    
    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A indices: 0,1,3,4 (row 0: 0,1; row 1: 3,4) -> idx = i*3+j
    # B indices: 0,1,3,4 (col 0: 0,1; col 1: 3,4) -> idx = j*3+k
    # C indices: 0,1,2,3 (row 0: 0,1; row 1: 2,3) -> idx = k*2+i (Trace: W[k*n+i])
    # Note: C is 2x3. C[0,0]=0, C[0,1]=1, C[0,2]=2? No.
    # Trace convention W[k*n+i, r].
    # C is n x p? No, C is n x p. n=2, p=3.
    # C[i, k]. idx = k*n + i.
    # C[0,0] -> 0*2+0=0
    # C[0,1] -> 1*2+0=2
    # C[0,2] -> 2*2+0=4
    # C[1,0] -> 0*2+1=1
    # C[1,1] -> 1*2+1=3
    # C[1,2] -> 2*2+1=5
    
    # Let's re-verify the indices in the current program.
    # Current program:
    # W[0, 0] = 1; W[3, 0] = 1  # M1 -> C00, C11
    # C00 is idx 0. C11 is idx 3.
    # My mapping: C[0,0]=0, C[1,1]=3. Correct.
    
    # M1: (A00+A11)(B00+B11)
    # A00=0, A11=4. B00=0, B11=4.
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (A10+A11)B00
    # A10=3, A11=4. B00=0.
    # -> C10 (idx 1), -C11 (idx 3)
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3: A00(B01-B11)
    # A00=0. B01=1, B11=4.
    # -> C01 (idx 2), C11 (idx 3)
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4: A11(B10-B00)
    # A11=4. B10=3, B00=0.
    # -> C00 (idx 0), C10 (idx 1)
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5: (A00+A01)B11
    # A00=0, A01=1. B11=4.
    # -> -C00 (idx 0), C01 (idx 2)
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6: (A10-A00)(B00+B01)
    # A10=3, A00=0. B00=0, B01=1.
    # -> C11 (idx 3)
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: (A01-A11)(B10+B11)
    # A01=1, A11=4. B10=3, B11=4.
    # -> C00 (idx 0)
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # Part 2: Schoolbook for A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    # A indices: 2, 5
    # B indices: 6, 7
    # C indices: 0, 1, 2, 3
    
    # r=7: A02 B20 -> C00
    U[2, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1
    
    # r=8: A02 B21 -> C01
    U[2, 8] = 1
    V[7, 8] = 1
    W[2, 8] = 1
    
    # r=9: A12 B20 -> C10
    U[5, 9] = 1
    V[6, 9] = 1
    W[1, 9] = 1
    
    # r=10: A12 B21 -> C11
    U[5, 10] = 1
    V[7, 10] = 1
    W[3, 10] = 1
    
    # Part 3: Schoolbook for A[0:2, 0:3] * B[0:3, 2] -> C[0:2, 2]
    # A indices: 0,1,2,3,4,5
    # B indices: 2,5,8
    # C indices: 4,5
    
    # r=11: A00 B02 -> C02
    U[0, 11] = 1
    V[2, 11] = 1
    W[4, 11] = 1
    
    # r=12: A01 B12 -> C02
    U[1, 12] = 1
    V[5, 12] = 1
    W[4, 12] = 1
    
    # r=13: A02 B22 -> C02
    U[2, 13] = 1
    V[8, 13] = 1
    W[4, 13] = 1
    
    # r=14: A10 B02 -> C12
    U[3, 14] = 1
    V[2, 14] = 1
    W[5, 14] = 1
    
    # r=15: A11 B12 -> C12
    U[4, 15] = 1
    V[5, 15] = 1
    W[5, 15] = 1
    
    # r=16: A12 B22 -> C12
    U[5, 16] = 1
    V[8, 16] = 1
    W[5, 16] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
