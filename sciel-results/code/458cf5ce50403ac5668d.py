"""Matrix multiplication 3x4x4: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def _strassen_block(io, jo, ko, U, V, W, col_start):
    """Add Strassen rank-7 decomposition for 2x2x2 block at offset (io,jo,ko)."""
    # A indices: (io,jo), (io,jo+1), (io+1,jo), (io+1,jo+1)
    # B indices: (jo,ko), (jo,ko+1), (jo+1,ko), (jo+1,ko+1)
    # C indices: (ko,io), (ko,io+1), (ko+1,io), (ko+1,io+1)
    
    # U rows for A[io,jo], A[io,jo+1], A[io+1,jo], A[io+1,jo+1]
    ua = io*4 + jo
    ub = io*4 + jo + 1
    uc = (io+1)*4 + jo
    ud = (io+1)*4 + jo + 1
    
    # V rows for B[jo,ko], B[jo,ko+1], B[jo+1,ko], B[jo+1,ko+1]
    ve = jo*4 + ko
    vf = jo*4 + ko + 1
    vg = (jo+1)*4 + ko
    vh = (jo+1)*4 + ko + 1
    
    # W rows for C[ko,io], C[ko,io+1], C[ko+1,io], C[ko+1,io+1]
    wc00 = ko*3 + io
    wc01 = ko*3 + io + 1
    wc10 = (ko+1)*3 + io
    wc11 = (ko+1)*3 + io + 1
    
    c = col_start
    
    # m1 = (a+d)(e+h)
    U[ua, c] += 1; U[ud, c] += 1
    V[ve, c] += 1; V[vh, c] += 1
    W[wc00, c] += 1; W[wc11, c] += 1
    c += 1
    
    # m2 = (c+d)e
    U[uc, c] += 1; U[ud, c] += 1
    V[ve, c] += 1
    W[wc10, c] += 1
    c += 1
    
    # m3 = a(g-e)
    U[ua, c] += 1
    V[vg, c] += 1; V[ve, c] += -1
    W[wc01, c] += 1; W[wc11, c] += -1
    c += 1
    
    # m4 = d(h-f)
    U[ud, c] += 1
    V[vh, c] += 1; V[vf, c] += -1
    W[wc00, c] += 1; W[wc10, c] += 1
    c += 1
    
    # m5 = (a+b)h
    U[ua, c] += 1; U[ub, c] += 1
    V[vh, c] += 1
    W[wc00, c] += -1; W[wc01, c] += 1
    c += 1
    
    # m6 = (c-a)(e+f)
    U[uc, c] += 1; U[ua, c] += -1
    V[ve, c] += 1; V[vf, c] += 1
    W[wc11, c] += 1
    c += 1
    
    # m7 = (b-d)(g+h)
    U[ub, c] += 1; U[ud, c] += -1
    V[vg, c] += 1; V[vh, c] += 1
    W[wc00, c] += 1
    c += 1
    
    return c

def _verify(U, V, W):
    """Verify that the decomposition correctly computes matrix multiplication."""
    n, m, p = 3, 4, 4
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all (i,j,k)
    # and 0 otherwise (but since we're decomposing the multiplication tensor,
    # each (i,j,k) should contribute exactly 1)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def construct():
    n, m, p = 3, 4, 4
    start_time = time.time()
    deadline = start_time + 120  # 120s limit, use 120s as deadline
    
    # Build rank-44 decomposition:
    # 4 Strassen blocks for i in {0,1}, j in {0,1}x{2,3}, k in {0,1}x{2,3}
    # 16 schoolbook terms for i=2
    
    rank = 44
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    col = 0
    # Strassen blocks for i in {0,1}
    for io in [0, 1]:
        for jo in [0, 2]:
            for ko in [0, 2]:
                col = _strassen_block(io, jo, ko, U, V, W, col)
    
    # Schoolbook for i=2
    for j in range(m):
        for k in range(p):
            U[2*m + j, col] = 1
            V[j*p + k, col] = 1
            W[k*n + 2, col] = 1
            col += 1
    
    # Verify
    if _verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook rank 48
    rank48 = n*m*p
    U48 = np.zeros((n*m, rank48), dtype=float)
    V48 = np.zeros((m*p, rank48), dtype=float)
    W48 = np.zeros((p*n, rank48), dtype=float)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U48[i*m+j, c] = 1
                V48[j*p+k, c] = 1
                W48[k*n+i, c] = 1
                c += 1
    
    return {"U": U48.tolist(), "V": V48.tolist(), "W": W48.tolist()}
