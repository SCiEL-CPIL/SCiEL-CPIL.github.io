import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # j=0 slice: 2x1x5, rank 10 (schoolbook)
    for i in range(n):
        for k in range(p):
            U[i*m + 0, col] = 1
            V[0*p + k, col] = 1
            W[k*n + i, col] = 1
            col += 1
    # col = 10
    
    # j=1,2, k=0,1: 2x2x2, rank 7 (Strassen)
    i_off, j_off, k_off = 0, 1, 0
    U[i_off*m + j_off, col] = 1
    U[(i_off+1)*m + (j_off+1), col] = 1
    V[j_off*p + k_off, col] = 1
    V[(j_off+1)*p + (k_off+1), col] = 1
    W[k_off*n + i_off, col] = 1
    W[(k_off+1)*n + (i_off+1), col] = 1
    col += 1
    
    U[(i_off+1)*m + j_off, col] = 1
    U[(i_off+1)*m + (j_off+1), col] = 1
    V[j_off*p + k_off, col] = 1
    W[(k_off+1)*n + i_off, col] = 1
    W[(k_off+1)*n + (i_off+1), col] = -1
    col += 1
    
    U[i_off*m + j_off, col] = 1
    V[j_off*p + (k_off+1), col] = 1
    V[(j_off+1)*p + (k_off+1), col] = -1
    W[(k_off+1)*n + i_off, col] = 1
    W[(k_off+1)*n + (i_off+1), col] = 1
    col += 1
    
    U[(i_off+1)*m + (j_off+1), col] = 1
    V[(j_off+1)*p + k_off, col] = 1
    V[j_off*p + k_off, col] = -1
    W[k_off*n + i_off, col] = 1
    W[k_off*n + (i_off+1), col] = 1
    col += 1
    
    U[i_off*m + j_off, col] = 1
    U[i_off*m + (j_off+1), col] = 1
    V[(j_off+1)*p + (k_off+1), col] = 1
    W[k_off*n + i_off, col] = -1
    W[(k_off+1)*n + i_off, col] = 1
    col += 1
    
    U[(i_off+1)*m + j_off, col] = 1
    U[i_off*m + j_off, col] = -1
    V[j_off*p + k_off, col] = 1
    V[j_off*p + (k_off+1), col] = 1
    W[(k_off+1)*n + (i_off+1), col] = 1
    col += 1
    
    U[i_off*m + (j_off+1), col] = 1
    U[(i_off+1)*m + (j_off+1), col] = -1
    V[(j_off+1)*p + k_off, col] = 1
    V[(j_off+1)*p + (k_off+1), col] = 1
    W[k_off*n + i_off, col] = 1
    col += 1
    # col = 17
    
    # j=1,2, k=2: 2x2x1, rank 4 (schoolbook)
    for i in range(n):
        for j in [1, 2]:
            U[i*m + j, col] = 1
            V[j*p + 2, col] = 1
            W[2*n + i, col] = 1
            col += 1
    # col = 21
    
    # j=1,2, k=3,4: 2x2x2, rank 7 (Strassen)
    i_off, j_off, k_off = 0, 1, 3
    U[i_off*m + j_off, col] = 1
    U[(i_off+1)*m + (j_off+1), col] = 1
    V[j_off*p + k_off, col] = 1
    V[(j_off+1)*p + (k_off+1), col] = 1
    W[k_off*n + i_off, col] = 1
    W[(k_off+1)*n + (i_off+1), col] = 1
    col += 1
    
    U[(i_off+1)*m + j_off, col] = 1
    U[(i_off+1)*m + (j_off+1), col] = 1
    V[j_off*p + k_off, col] = 1
    W[(k_off+1)*n + i_off, col] = 1
    W[(k_off+1)*n + (i_off+1), col] = -1
    col += 1
    
    U[i_off*m + j_off, col] = 1
    V[j_off*p + (k_off+1), col] = 1
    V[(j_off+1)*p + (k_off+1), col] = -1
    W[(k_off+1)*n + i_off, col] = 1
    W[(k_off+1)*n + (i_off+1), col] = 1
    col += 1
    
    U[(i_off+1)*m + (j_off+1), col] = 1
    V[(j_off+1)*p + k_off, col] = 1
    V[j_off*p + k_off, col] = -1
    W[k_off*n + i_off, col] = 1
    W[k_off*n + (i_off+1), col] = 1
    col += 1
    
    U[i_off*m + j_off, col] = 1
    U[i_off*m + (j_off+1), col] = 1
    V[(j_off+1)*p + (k_off+1), col] = 1
    W[k_off*n + i_off, col] = -1
    W[(k_off+1)*n + i_off, col] = 1
    col += 1
    
    U[(i_off+1)*m + j_off, col] = 1
    U[i_off*m + j_off, col] = -1
    V[j_off*p + k_off, col] = 1
    V[j_off*p + (k_off+1), col] = 1
    W[(k_off+1)*n + (i_off+1), col] = 1
    col += 1
    
    U[i_off*m + (j_off+1), col] = 1
    U[(i_off+1)*m + (j_off+1), col] = -1
    V[(j_off+1)*p + k_off, col] = 1
    V[(j_off+1)*p + (k_off+1), col] = 1
    W[k_off*n + i_off, col] = 1
    col += 1
    # col = 28
    
    assert col == rank
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
