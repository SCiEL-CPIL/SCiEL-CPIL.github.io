import time, random
import numpy as np

DIMENSIONS = [3, 4, 8]

def _verify(U, V, W, n, m, p, trials=6):
    Ua = np.array(U, dtype=object); Va = np.array(V, dtype=object); Wa = np.array(W, dtype=object)
    rng = random.Random(7)
    for _ in range(trials):
        A = np.array([[rng.randint(-3,3) for _ in range(m)] for _ in range(n)], dtype=object)
        B = np.array([[rng.randint(-3,3) for _ in range(p)] for _ in range(m)], dtype=object)
        C = A @ B
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    for r in range(len(Ua[0])):
                        s += Wa[k*n+i, r] * Ua[i*m+j, r] * Va[j*p+k, r] * A[i,j] * B[j,k]
                if s != C[i][k]:
                    return False
    return True

def _build(n, m, p, use_strassen):
    U = {r: [] for r in range(n*m)}
    V = {r: [] for r in range(m*p)}
    W = {r: [] for r in range(p*n)}
    def term(u, v, w):
        for r in range(n*m): U[r].append(u.get(r, 0))
        for r in range(m*p): V[r].append(v.get(r, 0))
        for r in range(p*n): W[r].append(w.get(r, 0))
    def ui(i,j): return i*m+j
    def vj(j,k): return j*p+k
    def wk(k,i): return k*n+i
    def strassen(i0,i1,j0,j1,k0,k1):
        a11,a12,a21,a22 = ui(i0,j0),ui(i0,j1),ui(i1,j0),ui(i1,j1)
        b11,b12,b21,b22 = vj(j0,k0),vj(j0,k1),vj(j1,k0),vj(j1,k1)
        c11,c12,c21,c22 = wk(k0,i0),wk(k1,i0),wk(k0,i1),wk(k1,i1)
        term({a11:1,a22:1},{b11:1,b22:1},{c11:1,c22:1})                      # p1
        term({a21:1,a22:1},{b11:1},{c21:1,c22:-1})                            # p2
        term({a11:1},{b21:1,b22:-1},{c12:1,c22:1})                            # p3
        term({a22:1},{b11:1,b21:-1},{c11:1,c21:1})                            # p4
        term({a11:1,a12:1},{b22:1},{c12:1,c11:-1})                            # p5
        term({a21:1,a11:-1},{b11:1,b12:1},{c22:1})                            # p6
        term({a12:1,a22:-1},{b21:1,b22:1},{c11:1})                            # p7
    def single(i,j0,j1,k0,k1):
        term({ui(i,j0):1},{vj(j0,k0):1},{wk(k0,i):1})
        term({ui(i,j1):1},{vj(j1,k0):1},{wk(k0,i):1})
        term({ui(i,j0):1},{vj(j0,k1):1},{wk(k1,i):1})
        term({ui(i,j1):1},{vj(j1,k1):1},{wk(k1,i):1})
    if use_strassen:
        strassen(0,1,0,1,0,1); strassen(0,1,2,3,0,1)
        strassen(0,1,0,1,2,3); strassen(0,1,2,3,2,3)
        single(2,0,1,0,1); single(2,2,3,0,1)
        single(2,0,1,2,3); single(2,2,3,2,3)
    else:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    term({ui(i,j):1},{vj(j,k):1},{wk(k,i):1})
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = 115.0
    def to_list(d, size):
        return [d[r] for r in range(size)]
    U,V,W = _build(n,m,p,use_strassen=True)
    if _verify(U,V,W,n,m,p) and time.time()-t0 < deadline:
        return {"U": to_list(U,n*m), "V": to_list(V,m*p), "W": to_list(W,p*n)}
    U,V,W = _build(n,m,p,use_strassen=False)
    return {"U": to_list(U,n*m), "V": to_list(V,m*p), "W": to_list(W,p*n)}
