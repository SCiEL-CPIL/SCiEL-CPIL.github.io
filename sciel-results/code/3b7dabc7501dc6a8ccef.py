import numpy as np

def construct():
    n, m, p = 3, 4, 5
    # Target rank 56
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen 2x2x2
    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1 = (A11+A22)(B11+B22)
        U[i0*4 + j0][col] += 1
        U[i0*4 + j0+1][col] += 1
        V[j0*5 + k0][col] += 1
        V[j0*5 + k0+1][col] += 1
        W[k0*3 + i0][col] += 1
        W[k0*3 + i0+1][col] += 1
        W[k0*3 + i0][col] -= 1 # Wait, C11 = M1...
        # Let's do it carefully.
        # C11 gets M1, C22 gets M1.
        # C12 gets -M5, C21 gets M4.
        
        # This manual addition is error prone. 
        # Better to define the terms explicitly.
