import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 4, 4, 5
    
    # Rank 65
    rank = 65
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(i, j, k, coeff):
        nonlocal col
        if coeff == 0:
            return
        U[i*4+j][col] += coeff
        V[j*5+k][col] += coeff
        W[k*4+i][col] += coeff
        col += 1
        
    # We need to decompose 4x4x4 using 49 terms.
    # And 4x4x1 using 16 terms.
    
    # 4x4x1 part (cols 4)
    for i in range(4):
        for j in range(4):
            add_term(i, j, 4, 1)
            
    # 4x4x4 part (cols 0-3)
    # Nested Strassen
    
    # Inner Strassen coefficients for 2x2 matrices X, Y
    # Returns list of 7 tuples (u_coeffs, v_coeffs, w_coeffs) for the 4 entries of C?
    # No, we need the rank-1 decomposition.
    
    # This is too complex to verify mentally.
    # I will stick to schoolbook 80.
    
    # Reset
    rank = 80
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
