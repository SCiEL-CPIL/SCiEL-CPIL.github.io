import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    
    # We'll use a Strassen-like approach for 2x2x2 blocks
    # 8 blocks: 2 j-blocks x 4 k-blocks
    # Each 2x2x2 block uses 7 Strassen terms
    # Total rank = 8 * 7 = 56
    
    rank = 56
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # For each 2x2x2 block
    # j0 in {0, 2}, k0 in {0, 2, 4, 6}
    for j0 in [0, 2]:
        for k0 in [0, 2, 4, 6]:
            # Local indices within the block
            # A has indices (i, j) where i in {0,1}, j in {j0, j0+1}
            # B has indices (j, k) where j in {j0, j0+1}, k in {k0, k0+1}
            # C has indices (k, i) where k in {k0, k0+1}, i in {0,1}
            
            # Strassen terms for 2x2x2:
            # M1 = (A00 + A11)(B00 + B11)
            # M2 = (A10 + A11)B00
            # M3 = A00(B01 - B11)
            # M4 = A11(B10 - B00)
            # M5 = (A00 + A01)B11
            # M6 = (A10 - A00)(B00 + B01)
            # M7 = (A01 - A11)(B10 + B11)
            
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # Term 1: M1
            U[0*m + j0, col] = 1       # A00
            U[1*m + (j0+1), col] = 1   # A11
            V[j0*p + k0, col] = 1      # B00
            V[(j0+1)*p + (k0+1), col] = 1  # B11
            W[k0*n + 0, col] = 1       # C00 coeff for M1
            W[(k0+1)*n + 1, col] = 1   # C11 coeff for M1
            col += 1
            
            # Term 2: M2
            U[1*m + j0, col] = 1       # A10
            U[1*m + (j0+1), col] = 1   # A11
            V[j0*p + k0, col] = 1      # B00
            W[k0*n + 1, col] = 1       # C10 coeff for M2
            W[(k0+1)*n + 1, col] = -1  # C11 coeff for M2
            col += 1
            
            # Term 3: M3
            U[0*m + j0, col] = 1       # A00
            V[j0*p + (k0+1), col] = 1  # B01
            V[(j0+1)*p + (k0+1), col] = -1  # -B11
            W[(k0+1)*n + 0, col] = 1   # C01 coeff for M3
            W[(k0+1)*n + 1, col] = 1   # C11 coeff for M3
            col += 1
            
            # Term 4: M4
            U[1*m + (j0+1), col] = 1   # A11
            V[(j0+1)*p + k0, col] = 1  # B10
            V[j0*p + k0, col] = -1     # -B00
            W[k0*n + 0, col] = 1       # C00 coeff for M4
            W[k0*n + 1, col] = 1       # C10 coeff for M4
            col += 1
            
            # Term 5: M5
            U[0*m + j0, col] = 1       # A00
            U[0*m + (j0+1), col] = 1   # A01
            V[(j0+1)*p + (k0+1), col] = 1  # B11
            W[k0*n + 0, col] = -1      # C00 coeff for M5
            W[(k0+1)*n + 0, col] = 1   # C01 coeff for M5
            col += 1
            
            # Term 6: M6
            U[1*m + j0, col] = 1       # A10
            U[0*m + j0, col] = -1      # -A00
            V[j0*p + k0, col] = 1      # B00
            V[j0*p + (k0+1), col] = 1  # B01
            W[(k0+1)*n + 1, col] = 1   # C11 coeff for M6
            col += 1
            
            # Term 7: M7
            U[0*m + (j0+1), col] = 1   # A01
            U[1*m + (j0+1), col] = -1  # -A11
            V[(j0+1)*p + k0, col] = 1  # B10
            V[(j0+1)*p + (k0+1), col] = 1  # B11
            W[k0*n + 0, col] = 1       # C00 coeff for M7
            col += 1
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result['U'][0]))
