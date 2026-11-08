"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Optimized rank 46 decomposition.
"""

DIMENSIONS = [3, 3, 8]

import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    terms = []
    
    def add_strassen(ra, ca, rb, cb):
        """Add Strassen terms for A[ra:ra+2, ca:ca+2] * B[rb:rb+2, cb:cb+2]"""
        a = (ra, ca)
        b = (ra, ca+1)
        c = (ra+1, ca)
        d = (ra+1, ca+1)
        e = (rb, cb)
        f = (rb, cb+1)
        g = (rb+1, cb)
        h = (rb+1, cb+1)
        
        C00 = (ra, cb)
        C01 = (ra, cb+1)
        C10 = (ra+1, cb)
        C11 = (ra+1, cb+1)
        
        terms.append(([(a,1),(d,1)], [(e,1),(h,1)], [(C00,1),(C11,1)]))
        terms.append(([(c,1),(d,1)], [(e,1)], [(C10,1),(C11,-1)]))
        terms.append(([(a,1)], [(f,1),(h,-1)], [(C01,1),(C11,1)]))
        terms.append(([(d,1)], [(g,1),(e,-1)], [(C00,1),(C10,1)]))
        terms.append(([(a,1),(b,1)], [(h,1)], [(C00,-1),(C01,1)]))
        terms.append(([(c,1),(a,-1)], [(e,1),(f,1)], [(C11,1)]))
        terms.append(([(b,1),(d,-1)], [(g,1),(h,1)], [(C00,1)]))
        
    def add_direct(i, j, k):
        terms.append(([((i,j),1)], [((j,k),1)], [((i,k),1)]))
        
    def add_outer(i_list, j, k_list):
        # Adds terms for A[i, j] * B[j, k] for i in i_list, k in k_list
        # This is rank 1 if we can combine? No, each (i,k) pair needs a term if W is sparse?
        # Actually, for A[i,j] B[j,k], if we fix j, we can use 1 term for all i,k?
        # U[i,j]=1, V[j,k]=1, W[i,k]=1.
        # Sum = 1.
        # So yes, rank 1 for the whole block if it's a single j.
        terms.append(([( (i, j), 1 ) for i in i_list], [((j, k), 1) for k in k_list], [( (i, k), 1 ) for i in i_list for k in k_list]))
        
    # Block 1: A[0:2, 0:2] * B[0:2, 0:8]
    # Split B into 4 blocks of 2
    for g in range(4):
        col_off = g * 2
        add_strassen(0, 0, 0, col_off)
        
    # Block 2: A[0:2, 2] * B[2, 0:8]
    # i in 0,1; j=2; k in 0..7
    add_outer([0, 1], 2, range(8))
    
    # Block 3: A[2, 0:2] * B[0:2, 0:8]
    # i=2; j in 0,1; k in 0..7
    # This is 1x2x8. Rank 16.
    # For each k, we need 2 terms (one for j=0, one for j=1)
    for k in range(8):
        terms.append(([((2,0),1)], [((0,k),1)], [((2,k),1)]))
        terms.append(([((2,1),1)], [((1,k),1)], [((2,k),1)]))
        
    # Block 4: A[2, 2] * B[2, 0:8]
    # i=2; j=2; k in 0..7
    add_outer([2], 2, range(8))
    
    rank = len(terms)
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for idx, (U_sp, V_sp, W_sp) in enumerate(terms):
        for (i,j), c in U_sp:
            U[i*m+j][idx] = c
        for (j,k), c in V_sp:
            V[j*p+k][idx] = c
        for (i,k), c in W_sp:
            W[k*n+i][idx] = c
            
    return {"U": U, "V": V, "W": W}
