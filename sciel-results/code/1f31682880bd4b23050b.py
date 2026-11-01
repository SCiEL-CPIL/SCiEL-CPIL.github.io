"""Matrix multiplication 2,2,12 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
    
    Uses a block-diagonal construction: split the 2x12 matrix into 6 independent 2x2 blocks
    and apply a rank-7 Strassen-type decomposition to each block.
    Total rank = 6 * 7 = 42.
    
    The decomposition satisfies:
    For all i in [0,2), j in [0,2), k in [0,12):
      sum_c U[i*2+j, c] * V[j*12+k, c] * W[k*2+i, c] = 1
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    num_blocks = p // 2  # 6
    rank = num_blocks * 7  # 42
    
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    # Strassen's 7 scalar multiplications for 2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    for t in range(num_blocks):
        base = t * 7
        k0 = 2 * t      # first column in block
        k1 = 2 * t + 1  # second column in block
        
        for s in range(7):
            c = base + s
            
            if s == 0:
                # M1 = (A00+A11)*(B00+B11)
                U[0, c] += 1
                U[3, c] += 1
                V[0*p + k0, c] += 1
                V[1*p + k1, c] += 1
                W[k0*n + 0, c] += 1
                W[k1*n + 1, c] += 1
                
            elif s == 1:
                # M2 = (A10+A11)*B00
                U[2, c] += 1
                U[3, c] += 1
                V[0*p + k0, c] += 1
                W[k0*n + 1, c] += 1
                W[k1*n + 1, c] -= 1
                
            elif s == 2:
                # M3 = A00*(B01-B11)
                U[0, c] += 1
                V[0*p + k1, c] += 1
                V[1*p + k1, c] -= 1
                W[k1*n + 0, c] += 1
                W[k1*n + 1, c] += 1
                
            elif s == 3:
                # M4 = A11*(B10-B00)
                U[3, c] += 1
                V[1*p + k0, c] += 1
                V[0*p + k0, c] -= 1
                W[k0*n + 0, c] += 1
                W[k0*n + 1, c] += 1
                
            elif s == 4:
                # M5 = (A00+A01)*B11
                U[0, c] += 1
                U[1, c] += 1
                V[1*p + k1, c] += 1
                W[k0*n + 0, c] -= 1
                W[k1*n + 0, c] += 1
                
            elif s == 5:
                # M6 = (A10-A00)*(B00+B01)
                U[2, c] += 1
                U[0, c] -= 1
                V[0*p + k0, c] += 1
                V[0*p + k1, c] += 1
                W[k1*n + 1, c] += 1
                
            elif s == 6:
                # M7 = (A01-A11)*(B10+B11)
                U[1, c] += 1
                U[3, c] -= 1
                V[1*p + k0, c] += 1
                V[1*p + k1, c] += 1
                W[k0*n + 0, c] += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
