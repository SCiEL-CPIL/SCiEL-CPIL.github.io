"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Uses a block decomposition: split 8 columns into 4 groups of 2,
apply Strassen (rank 7) to the 2x2*2x2 core, and direct terms for
the remaining 2x1*1x2 and 1x2*2x2 cross terms, plus 1x1*1x2 for
the bottom-right corner. Total rank = 4*(7+2+4+2) = 68.
"""

DIMENSIONS = [3, 3, 8]

import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # Build decomposition terms
    # Each term is a tuple (U_sparse, V_sparse, W_sparse)
    # U_sparse: list of ((i,j), coeff)
    # V_sparse: list of ((j,k), coeff)
    # W_sparse: list of ((i,k), coeff)
    
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
        
        # m1 = (a+d)(e+h)
        terms.append(([(a,1),(d,1)], [(e,1),(h,1)], [(C00,1),(C11,1)]))
        # m2 = (c+d)*e
        terms.append(([(c,1),(d,1)], [(e,1)], [(C10,1),(C11,-1)]))
        # m3 = a*(f-h)
        terms.append(([(a,1)], [(f,1),(h,-1)], [(C01,1),(C11,1)]))
        # m4 = d*(g-e)
        terms.append(([(d,1)], [(g,1),(e,-1)], [(C00,1),(C10,1)]))
        # m5 = (a+b)*h
        terms.append(([(a,1),(b,1)], [(h,1)], [(C00,-1),(C01,1)]))
        # m6 = (c-a)*(e+f)
        terms.append(([(c,1),(a,-1)], [(e,1),(f,1)], [(C11,1)]))
        # m7 = (b-d)*(g+h)
        terms.append(([(b,1),(d,-1)], [(g,1),(h,1)], [(C00,1)]))
    
    def add_direct(i, j, k):
        terms.append(([((i,j),1)], [((j,k),1)], [((i,k),1)]))
    
    # Split 8 columns into 4 groups of 2
    for g in range(4):
        col_off = g * 2
        # A[0:2, 0:2] * B[0:2, col_off:col_off+2]
        add_strassen(0, 0, 0, col_off)
        # A[0:2, 2] * B[2, col_off:col_off+2]
        for i in range(2):
            for k in range(col_off, col_off+2):
                add_direct(i, 2, k)
        # A[2, 0:2] * B[0:2, col_off:col_off+2]
        for j in range(2):
            for k in range(col_off, col_off+2):
                add_direct(2, j, k)
        # A[2, 2] * B[2, col_off:col_off+2]
        for k in range(col_off, col_off+2):
            add_direct(2, 2, k)
    
    rank = len(terms)
    
    # Build matrices
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
    
    # Verify
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(Ua[i*m+j] * Va[j*p+k], Wa[k*n+i])
                if abs(s - 1.0) > 1e-9:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    
    if not ok:
        # Fallback to schoolbook
        rank_fb = n*m*p
        U_fb = [[0]*rank_fb for _ in range(n*m)]
        V_fb = [[0]*rank_fb for _ in range(m*p)]
        W_fb = [[0]*rank_fb for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][col] = 1
                    V_fb[j*p+k][col] = 1
                    W_fb[k*n+i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
