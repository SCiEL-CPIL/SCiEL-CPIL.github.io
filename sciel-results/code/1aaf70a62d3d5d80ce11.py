import time
import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 2x2 matrix 
    by a 2x3 matrix.
    
    The rank of M_{2,2,3} is 11.
    We construct the solution using Strassen for the 2x2 part (rank 7) 
    and schoolbook for the 2x1 part (rank 4).
    """
    n, m, p = 2, 2, 3
    rank = 11
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def set_u(i, j, r, val): U[i*m+j][r] = val
    def set_v(j, k, r, val): V[j*p+k][r] = val
    def set_w(k, i, r, val): W[k*n+i][r] = val
    
    # Strassen terms for k=0,1 (indices 0-6)
    # M1 = A00*(B01+B11)
    set_u(0, 0, 0, 1)
    set_v(0, 1, 0, 1)
    set_v(1, 1, 0, 1)
    set_w(1, 0, 0, 1) # C01
    set_w(1, 1, 0, 1) # C11
    
    # M2 = A10*(B00+B01)
    set_u(1, 0, 1, 1)
    set_v(0, 0, 1, 1)
    set_v(0, 1, 1, 1)
    set_w(0, 1, 1, 1) # C10
    set_w(1, 1, 1, 1) # C11
    
    # M3 = A00*A11
    set_u(0, 0, 2, 1)
    set_u(1, 1, 2, 1)
    set_v(0, 0, 2, 1)
    set_v(1, 1, 2, 1)
    set_w(0, 0, 2, 1) # C00
    set_w(1, 1, 2, 1) # C11
    
    # M4 = A11*(B10-B00)
    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1)
    set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1) # C00
    set_w(0, 1, 3, 1) # C10
    
    # M5 = A01*(B00+B11)
    set_u(0, 1, 4, 1)
    set_v(0, 0, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, -1) # C00
    set_w(1, 0, 4, 1) # C01
    
    # M6 = (A10-A00)*(B11-B10)
    set_u(1, 0, 5, 1)
    set_u(0, 0, 5, -1)
    set_v(1, 1, 5, 1)
    set_v(1, 0, 5, -1)
    set_w(1, 1, 5, 1) # C11
    
    # M7 = (A01-A11)*(B01-B10)
    set_u(0, 1, 6, 1)
    set_u(1, 1, 6, -1)
    set_v(0, 1, 6, 1)
    set_v(1, 0, 6, -1)
    set_w(0, 0, 6, 1) # C00
    
    # Schoolbook terms for k=2 (indices 7-10)
    # C02 = A00*B02 + A01*B12
    # C12 = A10*B02 + A11*B12
    
    # Term 7: A00*B02 -> contributes to C02
    set_u(0, 0, 7, 1)
    set_v(0, 2, 7, 1)
    set_w(2, 0, 7, 1)
    
    # Term 8: A01*B12 -> contributes to C02
    set_u(0, 1, 8, 1)
    set_v(1, 2, 8, 1)
    set_w(2, 0, 8, 1)
    
    # Term 9: A10*B02 -> contributes to C12
    set_u(1, 0, 9, 1)
    set_v(0, 2, 9, 1)
    set_w(2, 1, 9, 1)
    
    # Term 10: A11*B12 -> contributes to C12
    set_u(1, 1, 10, 1)
    set_v(1, 2, 10, 1)
    set_w(2, 1, 10, 1)
    
    return {"U": U, "V": V, "W": W}
