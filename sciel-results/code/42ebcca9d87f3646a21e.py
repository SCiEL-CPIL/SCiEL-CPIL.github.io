"""Program to find a rank-minimal tensor decomposition for 2x3 by 3x5 matrix multiplication."""
import time
import random

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 15
    pn = p * n  # 10
    # The schoolbook rank is n*m*p = 30. We start with this as the baseline.
    
    # A rank-30 decomposition is trivial and guaranteed.
    # U is 6x30, V is 15x30, W is 10x30
    rank = nm * mp // mp * p  # Actually n*m*p = 30
    
    # Initialize with the standard basis for the 30 dimensions
    U = [[0.0] * rank for _ in range(nm)]
    V = [[0.0] * rank for _ in range(mp)]
    W = [[0.0] * rank for _ in range(pn)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1.0
                V[j*p + k][col] = 1.0
                W[k*n + i][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Just to ensure it runs without error
    res = construct()
    print("Keys:", res.keys())
