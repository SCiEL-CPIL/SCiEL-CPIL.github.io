"""Tensor decomposition for 2x2 x 2x16 matrix multiplication using Strassen blocks."""
import time

DIMENSIONS = [2, 2, 16]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x2 * 2x16 matrix multiplication.
    
    Uses 8 independent Strassen blocks (one per pair of output columns), giving rank 8*7=56.
    """
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = 2, 2, 16
    rank = 56  # 8 Strassen blocks * 7 terms each
    
    # Strassen's 7 terms for 2x2*2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B10)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    #
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For our case, A is 2x2, B is 2x16.
    # Split B into 8 blocks of 2 columns each: B = [B^(0) B^(1) ... B^(7)]
    # Each B^(s) is 2x2.
    # C = A*B gives 2x16, split into C = [C^(0) C^(1) ... C^(7)]
    # Each C^(s) = A * B^(s) can be computed with Strassen using 7 multiplications.
    
    # U: shape (n*m, rank) = (4, 56)
    # V: shape (m*p, rank) = (32, 56)
    # W: shape (p*n, rank) = (32, 56)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for U (A factors):
    # M1: A00+A11 -> U[0,0]=1, U[3,0]=1 (row i*2+j: A00=row0,col0; A11=row1,col1)
    # M2: A10+A11 -> U[2,0]=1, U[3,0]=1 (A10=row1,col0; A11=row1,col1)
    # M3: A00 -> U[0,0]=1
    # M4: A11 -> U[3,0]=1
    # M5: A00+A01 -> U[0,0]=1, U[1,0]=1 (A00=row0,col0; A01=row0,col1)
    # M6: A10-A00 -> U[2,0]=1, U[0,0]=-1
    # M7: A01-A11 -> U[1,0]=1, U[3,0]=-1
    
    # Strassen coefficients for V (B factors per 2x2 block):
    # M1: B00+B11 -> V[0,0]=1, V[3,0]=1 (B00=row0,col0; B11=row1,col1)
    # M2: B00 -> V[0,0]=1
    # M3: B10-B11 -> V[2,0]=1, V[3,0]=-1 (B10=row1,col0; B11=row1,col1)
    # M4: B01-B10 -> V[1,0]=1, V[2,0]=-1 (B01=row0,col1; B10=row1,col0)
    # M5: B11 -> V[3,0]=1
    # M6: B00+B10 -> V[0,0]=1, V[2,0]=1
    # M7: B01+B11 -> V[1,0]=1, V[3,0]=1
    
    # Strassen coefficients for W (C output):
    # C00 = M1 - M4 + M5 + M7 -> W[0,0]=1, W[0,3]=-1, W[0,4]=1, W[0,6]=1
    # C01 = M3 + M5 -> W[1,2]=1, W[1,4]=1
    # C10 = M2 + M4 -> W[2,1]=1, W[2,3]=1
    # C11 = M1 - M2 + M3 + M6 -> W[3,0]=1, W[3,1]=-1, W[3,2]=1, W[3,5]=1
    
    # U coefficients per Strassen term t=0..6 for each (i,j) in A:
    # t=0 (M1): A00+A11
    # t=1 (M2): A10+A11
    # t=2 (M3): A00
    # t=3 (M4): A11
    # t=4 (M5): A00+A01
    # t=5 (M6): A10-A00
    # t=6 (M7): A01-A11
    
    U_coeffs = {
        # (i,j): [coeff for t=0,1,2,3,4,5,6]
        (0,0): [1, 0, 1, 0, 1, -1, 0],  # A00
        (0,1): [0, 0, 0, 0, 1, 0, 1],   # A01
        (1,0): [0, 1, 0, 0, 0, 1, 0],   # A10
        (1,1): [1, 1, 0, 1, 0, 0, -1],  # A11
    }
    
    # V coefficients per Strassen term t=0..6 for each (j,k_local) in B block:
    # k_local = 0 or 1 within each 2-column block
    # t=0 (M1): B00+B11
    # t=1 (M2): B00
    # t=2 (M3): B10-B11
    # t=3 (M4): B01-B10
    # t=4 (M5): B11
    # t=5 (M6): B00+B10
    # t=6 (M7): B01+B11
    
    V_coeffs = {
        # (j, k_local): [coeff for t=0,1,2,3,4,5,6]
        (0,0): [1, 1, 0, 0, 0, 1, 0],   # B00
        (0,1): [0, 0, 0, 1, 0, 0, 1],   # B01
        (1,0): [0, 0, 1, -1, 0, 1, 0],  # B10
        (1,1): [1, 0, -1, 0, 1, 0, 1],  # B11
    }
    
    # W coefficients per Strassen term t=0..6 for each (k_local, i) in C block:
    # W[k*n+i] where k is global col index, i is row
    # For block s, columns are 2s and 2s+1
    # C00 = M1 - M4 + M5 + M7 -> (i=0, k_local=0): t=0:+1, t=3:-1, t=4:+1, t=6:+1
    # C01 = M3 + M5 -> (i=0, k_local=1): t=2:+1, t=4:+1
    # C10 = M2 + M4 -> (i=1, k_local=0): t=1:+1, t=3:+1
    # C11 = M1 - M2 + M3 + M6 -> (i=1, k_local=1): t=0:+1, t=1:-1, t=2:+1, t=5:+1
    
    W_coeffs = {
        # (k_local, i): [coeff for t=0,1,2,3,4,5,6]
        (0,0): [1, 0, 0, -1, 1, 0, 1],   # C00
        (1,0): [0, 0, 1, 0, 1, 0, 0],    # C01
        (0,1): [0, 1, 0, 1, 0, 0, 0],    # C10
        (1,1): [1, -1, 1, 0, 0, 1, 0],   # C11
    }
    
    # Build the matrices
    for s in range(8):  # 8 blocks
        for t in range(7):  # 7 Strassen terms per block
            r = s * 7 + t  # rank index
            
            # Fill U
            for (i, j), coeffs in U_coeffs.items():
                U[i * m + j][r] = coeffs[t]
            
            # Fill V: for block s, columns are 2s and 2s+1
            for (j, k_local), coeffs in V_coeffs.items():
                k_global = 2 * s + k_local
                V[j * p + k_global][r] = coeffs[t]
            
            # Fill W: for block s, columns are 2s and 2s+1
            for (k_local, i), coeffs in W_coeffs.items():
                k_global = 2 * s + k_local
                W[k_global * n + i][r] = coeffs[t]
    
    if time.time() > deadline:
        # Should not happen, but return what we have
        pass
    
    return {"U": U, "V": V, "W": W}
