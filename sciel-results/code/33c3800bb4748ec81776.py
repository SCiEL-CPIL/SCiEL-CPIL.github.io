"""Matrix multiplication 3,4,6 - Rank-minimal tensor decomposition using Strassen blocks."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    n, m, p = 3, 4, 6
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_basis_term(i, j, k):
        """Add a single rank-1 basis term: e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}"""
        u = [0.0] * (n * m)
        v = [0.0] * (m * p)
        w = [0.0] * (p * n)
        u[i * m + j] = 1.0
        v[j * p + k] = 1.0
        w[k * n + i] = 1.0
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    def add_strassen_2x2x2(i0, j0, k0):
        """
        Add 7 rank-1 terms implementing Strassen's algorithm for the 2x2x2
        sub-tensor covering A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2].
        
        The 7 terms correspond to Strassen's 7 products:
        M1 = (A00+A11)(B00+B11)
        M2 = (A10+A11)B00  
        M3 = A00(B10-B11)
        M4 = A11(B01-B00)
        M5 = (A00+A01)B11
        M6 = (A10-A00)(B00+B01)
        M7 = (A01-A11)(B10+B11)
        """
        i0, i1 = i0, i0 + 1
        j0, j1 = j0, j0 + 1
        k0, k1 = k0, k0 + 1
        
        # M1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
        # u: +1 at (i0,j0), +1 at (i1,j1)
        # v: +1 at (j0,k0), +1 at (j1,k1)
        # w: +1 at (k0,i0), +1 at (k1,i1)
        u = [0.0]*(n*m); u[i0*m+j0]=1; u[i1*m+j1]=1
        v = [0.0]*(m*p); v[j0*p+k0]=1; v[j1*p+k1]=1
        w = [0.0]*(p*n); w[k0*n+i0]=1; w[k1*n+i1]=1
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
        # M2 = (A[i1,j0]+A[i1,j1]) * B[j0,k0]
        # u: +1 at (i1,j0), +1 at (i1,j1)
        # v: +1 at (j0,k0)
        # w: +1 at (k0,i1)
        u = [0.0]*(n*m); u[i1*m+j0]=1; u[i1*m+j1]=1
        v = [0.0]*(m*p); v[j0*p+k0]=1
        w = [0.0]*(p*n); w[k0*n+i1]=1
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
        # M3 = A[i0,j0] * (B[j1,k0]-B[j1,k1])
        # u: +1 at (i0,j0)
        # v: +1 at (j1,k0), -1 at (j1,k1)
        # w: +1 at (k0,i0)
        u = [0.0]*(n*m); u[i0*m+j0]=1
        v = [0.0]*(m*p); v[j1*p+k0]=1; v[j1*p+k1]=-1
        w = [0.0]*(p*n); w[k0*n+i0]=1
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
        # M4 = A[i1,j1] * (B[j0,k1]-B[j0,k0])
        # u: +1 at (i1,j1)
        # v: +1 at (j0,k1), -1 at (j0,k0)
        # w: +1 at (k1,i1)
        u = [0.0]*(n*m); u[i1*m+j1]=1
        v = [0.0]*(m*p); v[j0*p+k1]=1; v[j0*p+k0]=-1
        w = [0.0]*(p*n); w[k1*n+i1]=1
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
        # M5 = (A[i0,j0]+A[i0,j1]) * B[j1,k1]
        # u: +1 at (i0,j0), +1 at (i0,j1)
        # v: +1 at (j1,k1)
        # w: +1 at (k1,i0)
        u = [0.0]*(n*m); u[i0*m+j0]=1; u[i0*m+j1]=1
        v = [0.0]*(m*p); v[j1*p+k1]=1
        w = [0.0]*(p*n); w[k1*n+i0]=1
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
        # M6 = (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
        # u: +1 at (i1,j0), -1 at (i0,j0)
        # v: +1 at (j0,k0), +1 at (j0,k1)
        # w: +1 at (k0,i1)
        u = [0.0]*(n*m); u[i1*m+j0]=1; u[i0*m+j0]=-1
        v = [0.0]*(m*p); v[j0*p+k0]=1; v[j0*p+k1]=1
        w = [0.0]*(p*n); w[k0*n+i1]=1
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
        # M7 = (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1])
        # u: +1 at (i0,j1), -1 at (i1,j1)
        # v: +1 at (j1,k0), +1 at (j1,k1)
        # w: +1 at (k0,i0)
        u = [0.0]*(n*m); u[i0*m+j1]=1; u[i1*m+j1]=-1
        v = [0.0]*(m*p); v[j1*p+k0]=1; v[j1*p+k1]=1
        w = [0.0]*(p*n); w[k0*n+i0]=1
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
    
    def add_2x2x3(i0, j0, k0):
        """2x2x3 = 2x2x2 Strassen (7 terms) + 2x2x1 schoolbook (4 terms) = 11 terms"""
        add_strassen_2x2x2(i0, j0, k0)
        for di in range(2):
            for dj in range(2):
                add_basis_term(i0+di, j0+dj, k0+2)
    
    def add_1x4x6_schoolbook(i0):
        """1x4x6 schoolbook = 24 terms"""
        for j in range(4):
            for k in range(6):
                add_basis_term(i0, j, k)
    
    # Build 3x4x6 decomposition:
    # Top 2 rows (i=0,1): 2x4x6 via 4 sub-blocks of 2x2x3
    for j0 in [0, 2]:
        for k0 in [0, 3]:
            add_2x2x3(0, j0, k0)
    
    # Bottom 1 row (i=2): 1x4x6 schoolbook
    add_1x4x6_schoolbook(2)
    
    rank = len(U_cols)
    assert rank == 68, f"Expected rank 68, got {rank}"
    
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
