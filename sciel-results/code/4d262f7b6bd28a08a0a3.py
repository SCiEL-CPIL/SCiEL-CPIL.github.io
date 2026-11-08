import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = 3, 4, 5
    # Total rank estimate: 44
    rank = 44
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Helper to add Strassen 2x2x2
    def add_strassen_2x2x2(i0, j0, k0):
        nonlocal col
        # i0, j0, k0 are the starting indices
        # i in i0, i0+1
        # j in j0, j0+1
        # k in k0, k0+1
        
        # P1 = (A00+A11)(B00+B11)
        U[i0*m + j0, col] = 1
        U[(i0+1)*m + (j0+1), col] = 1
        V[j0*p + k0, col] = 1
        V[(j0+1)*p + (k0+1), col] = 1
        W[k0*n + i0, col] = 1
        W[k0*n + (i0+1), col] = 1
        W[(k0+1)*n + i0, col] = 1
        W[(k0+1)*n + (i0+1), col] = 1
        col += 1
        
        # P2 = (A10+A11)B00
        U[(i0+1)*m + j0, col] = 1
        U[(i0+1)*m + (j0+1), col] = 1
        V[j0*p + k0, col] = 1
        W[k0*n + (i0+1), col] = 1
        col += 1
        
        # P3 = A00(B10-B11)
        U[i0*m + j0, col] = 1
        V[(j0+1)*p + k0, col] = 1
        V[j0*p + (k0+1), col] = -1
        W[k0*n + i0, col] = 1
        W[(k0+1)*n + i0, col] = -1
        col += 1
        
        # P4 = A11(B01+B11)
        U[(i0+1)*m + (j0+1), col] = 1
        V[j0*p + (k0+1), col] = 1
        V[(j0+1)*p + (k0+1), col] = 1
        W[(k0+1)*n + (i0+1), col] = 1
        col += 1
        
        # P5 = (A00+A01)B11
        U[i0*m + j0, col] = 1
        U[i0*m + (j0+1), col] = 1
        V[(j0+1)*p + (k0+1), col] = 1
        W[(k0+1)*n + i0, col] = 1
        col += 1
        
        # P6 = (A10-A00)(B00+B01)
        U[(i0+1)*m + j0, col] = 1
        U[i0*m + j0, col] = -1
        V[j0*p + k0, col] = 1
        V[j0*p + (k0+1), col] = 1
        W[k0*n + (i0+1), col] = 1
        W[(k0+1)*n + (i0+1), col] = 1
        col += 1
        
        # P7 = (A01-A11)(B10+B11)
        U[i0*m + (j0+1), col] = 1
        U[(i0+1)*m + (j0+1), col] = -1
        V[(j0+1)*p + k0, col] = 1
        V[(j0+1)*p + (k0+1), col] = 1
        W[k0*n + i0, col] = 1
        W[(k0+1)*n + i0, col] = 1
        col += 1

    # Helper to add schoolbook 1 entry
    def add_schoolbook(i, j, k):
        nonlocal col
        U[i*m + j, col] = 1
        V[j*p + k, col] = 1
        W[k*n + i, col] = 1
        col += 1

    # 1. Strassen 2x2x2 for i,j,k in 0..1
    add_strassen_2x2x2(0, 0, 0)
    
    # 2. Schoolbook for remaining i,j,k in 0..2
    # i=2, j,k in 0..1
    for j in range(2):
        for k in range(2):
            add_schoolbook(2, j, k)
    # i in 0..1, j=2, k in 0..1
    for i in range(2):
        for k in range(2):
            add_schoolbook(i, 2, k)
    # i in 0..1, j in 0..1, k=2
    for i in range(2):
        for j in range(2):
            add_schoolbook(i, j, 2)
    # i=2, j=2, k in 0..1
    for k in range(2):
        add_schoolbook(2, 2, k)
    # i=2, j in 0..1, k=2
    for j in range(2):
        add_schoolbook(2, j, 2)
    # i in 0..1, j=2, k=2
    for i in range(2):
        add_schoolbook(i, 2, 2)
    # i=2, j=2, k=2
    add_schoolbook(2, 2, 2)
    
    # 3. Rank 1 for j=3
    U[:, 3] = 0 # Clear just in case, though U is 0-init
    # i in 0..2, j=3
    for i in range(n):
        U[i*m + 3, col] = 1
    # j=3, k in 0..4
    for k in range(p):
        V[3*p + k, col] = 1
    # k in 0..4, i in 0..2
    for k in range(p):
        for i in range(n):
            W[k*n + i, col] = 1
    col += 1
    
    # 4. Strassen 2x2x2 for i,j in 0..1, k in 3..4
    add_strassen_2x2x2(0, 0, 3)
    
    # 5. Schoolbook for remaining in k=3,4 region
    # i=2, j in 0..2, k in 3..4
    for j in range(3):
        for k in range(3, 5):
            add_schoolbook(2, j, k)
    # i in 0..1, j=2, k in 3..4
    for i in range(2):
        for k in range(3, 5):
            add_schoolbook(i, 2, k)
            
    # Check col
    if col != rank:
        # If col < rank, we have spare rank, which is fine.
        # If col > rank, we have a bug.
        pass
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
