import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 5]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == tensor T where T[i,j,k] = 1 for all i,j,k"""
    rank = len(U[0])
    U_arr = np.array([[float(x) for x in row] for row in U], dtype=float)
    V_arr = np.array([[float(x) for x in row] for row in V], dtype=float)
    W_arr = np.array([[float(x) for x in row] for row in W], dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                s = 0.0
                for r in range(rank):
                    s += U_arr[idx_u, r] * V_arr[idx_v, r] * W_arr[idx_w, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_strassen_2x2x2(n, m, p, j0, k0, U, V, W, r_start):
    """
    Build Strassen decomposition for 2x2x2 block:
    A rows 0,1; cols j0,j0+1
    B rows j0,j0+1; cols k0,k0+1
    C rows 0,1; cols k0,k0+1
    
    Returns new r after adding 7 terms.
    """
    r = r_start
    
    # M1 = (A[0,j0]+A[1,j0+1])(B[j0,k0]+B[j0+1,k0+1])
    U[0*m + j0][r] = 1
    U[1*m + j0+1][r] = 1
    V[j0*p + k0][r] = 1
    V[(j0+1)*p + k0+1][r] = 1
    W[k0*n + 0][r] = 1
    W[(k0+1)*n + 1][r] = 1
    r += 1
    
    # M2 = (A[1,j0]+A[1,j0+1])B[j0,k0]
    U[1*m + j0][r] = 1
    U[1*m + j0+1][r] = 1
    V[j0*p + k0][r] = 1
    W[k0*n + 1][r] = 1
    W[(k0+1)*n + 1][r] = -1
    r += 1
    
    # M3 = A[0,j0](B[j0+1,k0+1]-B[j0,k0+1])
    U[0*m + j0][r] = 1
    V[(j0+1)*p + k0+1][r] = 1
    V[j0*p + k0+1][r] = -1
    W[(k0+1)*n + 0][r] = 1
    W[(k0+1)*n + 1][r] = 1
    r += 1
    
    # M4 = A[1,j0+1](B[j0,k0]-B[j0+1,k0])
    U[1*m + j0+1][r] = 1
    V[j0*p + k0][r] = 1
    V[(j0+1)*p + k0][r] = -1
    W[k0*n + 0][r] = 1
    W[k0*n + 1][r] = 1
    r += 1
    
    # M5 = (A[0,j0]+A[0,j0+1])B[j0+1,k0+1]
    U[0*m + j0][r] = 1
    U[0*m + j0+1][r] = 1
    V[(j0+1)*p + k0+1][r] = 1
    W[k0*n + 0][r] = -1
    W[(k0+1)*n + 0][r] = 1
    r += 1
    
    # M6 = (A[1,j0]-A[0,j0])(B[j0,k0+1]+B[j0+1,k0+1])
    U[1*m + j0][r] = 1
    U[0*m + j0][r] = -1
    V[j0*p + k0+1][r] = 1
    V[(j0+1)*p + k0+1][r] = 1
    W[(k0+1)*n + 1][r] = 1
    r += 1
    
    # M7 = (A[0,j0+1]-A[1,j0+1])(B[j0+1,k0]+B[j0+1,k0+1])
    U[0*m + j0+1][r] = 1
    U[1*m + j0+1][r] = -1
    V[(j0+1)*p + k0][r] = 1
    V[(j0+1)*p + k0+1][r] = 1
    W[k0*n + 0][r] = 1
    r += 1
    
    return r

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: A[:,2] * B[2,:] (2x1 times 1x5) -> rank 10
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
    
    # Part 2: A[:,0:2] * B[0:2,4] (2x2 times 2x1) -> rank 4
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
    
    # Part 3: A[:,0:2] * B[0:2,0:2] (2x2 times 2x2) -> Strassen rank 7
    r = _build_strassen_2x2x2(n, m, p, 0, 0, U, V, W, r)
    
    # Part 4: A[:,0:2] * B[0:2,2:4] (2x2 times 2x2) -> Strassen rank 7
    r = _build_strassen_2x2x2(n, m, p, 0, 2, U, V, W, r)
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Decomposition verification failed")
    
    return {"U": U, "V": V, "W": W}
