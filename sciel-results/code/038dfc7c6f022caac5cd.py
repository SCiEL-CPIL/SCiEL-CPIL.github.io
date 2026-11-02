"""Rank-minimal exact tensor decomposition for 2x5 @ 5x5 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    def verify_fast(U, V, W, n, m, p, tol=1e-9):
        """Fast vectorized verification using numpy."""
        r = len(U[0])
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # Diagonal check: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i*m+j], V_arr[j*p+k]) @ W_arr[k*n+i]
                    if abs(s - 1.0) > tol:
                        return False
        # Off-diagonal check
        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    for k in range(p):
                        for k2 in range(p):
                            for i3 in range(n):
                                if j == j2 and k == k2 and i == i3:
                                    continue
                                s = np.dot(U_arr[i*m+j], V_arr[j2*p+k]) @ W_arr[k2*n+i3]
                                if abs(s) > tol:
                                    return False
        return True
    
    def verify(U, V, W, n, m, p, tol=1e-9):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        r = len(U[0])
        # Check diagonal: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > tol:
                        return False
        # Check off-diagonal
        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    for k in range(p):
                        for k2 in range(p):
                            for i3 in range(n):
                                if j == j2 and k == k2 and i == i3:
                                    continue
                                s = 0.0
                                for c in range(r):
                                    s += U[i*m+j][c] * V[j2*p+k][c] * W[k2*n+i3][c]
                                if abs(s) > tol:
                                    return False
        return True
    
    def schoolbook(n, m, p):
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
    
    # Try rank 25 with structured parameterization exploiting n=2
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # b[i] values
    b = [1, 1]  # for i=0,1
    
    for j0 in range(m):
        for k0 in range(p):
            c = j0 * p + k0
            for i in range(n):
                U[i * m + j0][c] = 1 / b[i]
                W[k0 * n + i][c] = b[i]
            V[j0 * p + k0][c] = 1
    
    # Verify
    if verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # If verification fails, try with different b values
    b = [1, -1]
    for j0 in range(m):
        for k0 in range(p):
            c = j0 * p + k0
            for i in range(n):
                U[i * m + j0][c] = 1 / b[i]
                W[k0 * n + i][c] = b[i]
            V[j0 * p + k0][c] = 1
    
    if verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W = schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
