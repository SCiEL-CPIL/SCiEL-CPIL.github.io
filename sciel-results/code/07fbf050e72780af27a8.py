import time
import numpy as np

DIMENSIONS = [2, 4, 8]
_START_TIME = time.time()
_DEADLINE = 120.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Rank 56 decomposition using Strassen's algorithm on 2x2x2 blocks.
    # The 2x4x8 multiplication is decomposed into 8 independent 2x2x2 multiplications.
    # Each 2x2x2 multiplication has rank 7.
    # Total rank = 8 * 7 = 56.
    
    rank = 56
    U = np.zeros((n * m, rank))  # 8 x 56
    V = np.zeros((m * p, rank))  # 32 x 56
    W = np.zeros((p * n, rank))  # 16 x 56
    
    # Helper to add a Strassen term for a 2x2 * 2x2 block
    def add_strassen(i0, j0, k0, base_col):
        # M1 = (A00 + A11)(B00 + B11)
        U[i0*m + j0, base_col] += 1
        U[(i0+1)*m + (j0+1), base_col] += 1
        V[j0*p + k0, base_col] += 1
        V[(j0+1)*p + (k0+1), base_col] += 1
        W[k0*n + i0, base_col] += 1
        W[(k0+1)*n + (i0+1), base_col] += 1
        base_col += 1
        
        # M2 = (A10 + A11)B00
        U[(i0+1)*m + j0, base_col] += 1
        U[(i0+1)*m + (j0+1), base_col] += 1
        V[j0*p + k0, base_col] += 1
        W[k0*n + (i0+1), base_col] += 1
        W[(k0+1)*n + (i0+1), base_col] -= 1
        base_col += 1
        
        # M3 = A00(B01 - B11)
        U[i0*m + j0, base_col] += 1
        V[j0*p + (k0+1), base_col] += 1
        V[(j0+1)*p + (k0+1), base_col] -= 1
        W[(k0+1)*n + i0, base_col] += 1
        W[(k0+1)*n + (i0+1), base_col] += 1
        base_col += 1
        
        # M4 = A11(B10 - B00)
        U[(i0+1)*m + (j0+1), base_col] += 1
        V[(j0+1)*p + k0, base_col] += 1
        V[j0*p + k0, base_col] -= 1
        W[k0*n + i0, base_col] += 1
        W[k0*n + (i0+1), base_col] += 1
        base_col += 1
        
        # M5 = (A00 + A01)B11
        U[i0*m + j0, base_col] += 1
        U[i0*m + (j0+1), base_col] += 1
        V[(j0+1)*p + (k0+1), base_col] += 1
        W[k0*n + i0, base_col] -= 1
        W[(k0+1)*n + i0, base_col] += 1
        base_col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[(i0+1)*m + j0, base_col] += 1
        U[i0*m + j0, base_col] -= 1
        V[j0*p + k0, base_col] += 1
        V[j0*p + (k0+1), base_col] += 1
        W[(k0+1)*n + (i0+1), base_col] += 1
        base_col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[i0*m + (j0+1), base_col] += 1
        U[(i0+1)*m + (j0+1), base_col] -= 1
        V[(j0+1)*p + k0, base_col] += 1
        V[(j0+1)*p + (k0+1), base_col] += 1
        W[k0*n + i0, base_col] += 1
        base_col += 1
        
        return base_col

    col = 0
    # Decompose into 4 blocks of 2x4x2.
    # Each 2x4x2 is two 2x2x2 multiplications.
    for k0 in [0, 2, 4, 6]:
        # First 2x2x2: A[0:2, 0:2] * B[0:2, k0:k0+2]
        col = add_strassen(0, 0, k0, col)
        # Second 2x2x2: A[0:2, 2:4] * B[2:4, k0:k0+2]
        col = add_strassen(0, 2, k0, col)
        
    if _time_left() < 5.0:
        # Safety margin
        pass

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result['U'][0]))
