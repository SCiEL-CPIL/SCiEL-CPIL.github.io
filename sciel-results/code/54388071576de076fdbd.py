"""Matrix multiplication 3,3,4 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS

    def verify(U_mat, V_mat, W_mat, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += Fraction(U_mat[i*m+j][r]) * Fraction(V_mat[j*p+k][r]) * Fraction(W_mat[k*n+i][r])
                    if s != Fraction(1):
                        return False
        return True

    # Schoolbook decomposition (rank 36)
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    # Try to find a lower-rank decomposition using structured search
    # Key insight: use the fact that matrix multiplication is bilinear
    # and try to find linear dependencies in the tensor.
    
    # For now, return schoolbook as a safe baseline
    U, V, W = schoolbook()
    rank = len(U[0])
    
    if verify(U, V, W, rank):
        return {"U": U, "V": V, "W": W}
    
    # Should never reach here
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
