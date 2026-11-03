import numpy as np
from itertools import product
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Helper: add a Strassen 2x2*2x2 block
    # A block: rows i in [i0, i0+1], cols j in [j0, j0+1]
    # B block: rows j in [j0, j0+1], cols k in [k0, k0+1]
    # C block: rows k in [k0, k0+1], cols i in [i0, i0+1]
    def add_strassen(i0, j0, k0):
        nonlocal col
        # A indices: i0*m+j0, i0*m+j0+1, (i0+1)*m+j0, (i0+1)*m+j0+1
        a = i0*m + j0
        b = i0*m + (j0+1)
        c = (i0+1)*m + j0
        d = (i0+1)*m + (j0+1)
        
        # B indices: j0*p+k0, j0*p+k0+1, (j0+1)*p+k0, (j0+1)*p+k0+1
        e = j0*p + k0
        f = j0*p + (k0+1)
        g = (j0+1)*p + k0
        h = (j0+1)*p + (k0+1)
        
        # C indices: k*n+i
        # C00: k0*n+i0, C01: k0*n+i0+1, C10: (k0+1)*n+i0, C11: (k0+1)*n+i0+1
        c00 = k0*n + i0
        c10 = k0*n + (i0+1)
        c01 = (k0+1)*n + i0
        c11 = (k0+1)*n + (i0+1)
        
        # M1 = (A00+A11)(B00+B11) -> C00 + C11
        U[a, col] = 1; U[d, col] = 1
        V[e, col] = 1; V[h, col] = 1
        W[c00, col] = 1; W[c11, col] = 1
        col += 1
        
        # M2 = (A10+A11)B00 -> C10 - C11
        U[c, col] = 1; U[d, col] = 1
        V[e, col] = 1
        W[c10, col] = 1; W[c11, col] = -1
        col += 1
        
        # M3 = A00(B01-B11) -> C01 + C11
        U[a, col] = 1
        V[f, col] = 1; V[h, col] = -1
        W[c01, col] = 1; W[c11, col] = 1
        col += 1
        
        # M4 = A11(B10-B00) -> C00 + C10
        U[d, col] = 1
        V[g, col] = 1; V[e, col] = -1
        W[c00, col] = 1; W[c10, col] = 1
        col += 1
        
        # M5 = (A00+A01)B11 -> -C00 + C01
        U[a, col] = 1; U[b, col] = 1
        V[h, col] = 1
        W[c00, col] = -1; W[c01, col] = 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01) -> C11
        U[c, col] = 1; U[a, col] = -1
        V[e, col] = 1; V[f, col] = 1
        W[c11, col] = 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11) -> C00
        U[b, col] = 1; U[d, col] = -1
        V[g, col] = 1; V[h, col] = 1
        W[c00, col] = 1
        col += 1
    
    # Block 1: A_L (2x2, i=0-1, j=0-1) * B_T1 (2x2, j=0-1, k=0-1) -> C[0:2, 0:2]
    add_strassen(i0=0, j0=0, k0=0)
    
    # Block 2: A_L (2x2, i=0-1, j=0-1) * B_T2 (2x2, j=0-1, k=2-3) -> C[0:2, 2:4]
    add_strassen(i0=0, j0=0, k0=2)
    
    # Block 3: A_L (2x2, i=0-1, j=0-1) * B_T3 (2x1, j=0-1, k=4) -> C[0:2, 4:5]
    # Schoolbook: 2*2*1 = 4 terms
    for i in range(2):
        for j in range(2):
            U[i*m+j, col] = 1
            V[j*p+4, col] = 1
            W[4*n+i, col] = 1
            col += 1
    
    # Block 4: A_R (2x1, i=0-1, j=2) * B_B (1x5, j=2, k=0-4) -> C[0:2, 0:5]
    # Schoolbook: 2*1*5 = 10 terms
    for i in range(2):
        for k in range(5):
            U[i*m+2, col] = 1
            V[2*p+k, col] = 1
            W[k*n+i, col] = 1
            col += 1
    
    # Verify col count
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
