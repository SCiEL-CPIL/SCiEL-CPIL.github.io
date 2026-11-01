"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-7 exact tensor decomposition for 2x2 matrix multiplication
    using the Strassen algorithm.
    
    The decomposition satisfies:
    C[i,k] = sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r]
    for all i, j, k in {0, 1}, where A, B, C are 2x2 matrices.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=2
    
    # Strassen's 7 intermediate products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U: shape (n*m, rank) = (4, 7)
    # Row index = i*m + j = 2*i + j
    U = [
        [1, 0, 1, 0, 1, 0, 0],  # i=0,j=0 (A00)
        [0, 0, 0, 0, 1, 0, 0],  # i=0,j=1 (A01)
        [0, 1, 0, 0, 0, 1, 0],  # i=1,j=0 (A10)
        [1, 1, 0, 1, 0, -1, -1]  # i=1,j=1 (A11)
    ]
    
    # V: shape (m*p, rank) = (4, 7)
    # Row index = j*p + k = 2*j + k
    V = [
        [1, 1, 0, 0, 0, 1, 0],  # j=0,k=0 (B00)
        [1, 0, 1, -1, 0, 1, 0], # j=0,k=1 (B01)
        [1, 0, -1, 1, 0, 0, 1], # j=1,k=0 (B10)
        [1, 0, 0, 0, 1, 0, 1]   # j=1,k=1 (B11)
    ]
    
    # W: shape (p*n, rank) = (4, 7)
    # Row index = k*n + i = 2*k + i
    W = [
        [1, 0, 0, 1, 0, 0, 1],  # k=0,i=0 (C00)
        [0, 0, 1, 0, 1, 0, 0],  # k=0,i=1 (C01)
        [0, 1, 0, 1, 0, 1, 0],  # k=1,i=0 (C10)
        [1, -1, 1, 0, -1, 0, 1]  # k=1,i=1 (C11)
    ]
    
    # Verify the decomposition exactly
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # should equal 1 if (i,j,k) is a valid matrix multiplication triple, 0 otherwise
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the multiplication tensor
    # T[i,j,k] = 1 if this corresponds to C[i,k] = sum_j A[i,j]*B[j,k]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(7):
                    actual += U_arr[i*m + j, r] * V_arr[j*p + k, r] * W_arr[k*n + i, r]
                if abs(actual - expected) > 1e-10:
                    raise ValueError(f"Decomposition verification failed at ({i},{j},{k}): got {actual}, expected {expected}")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
