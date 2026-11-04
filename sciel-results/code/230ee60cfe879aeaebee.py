"""Tensor decomposition search for 2x3x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def _validate(U, V, W, n, m, p, seed, n_tests=6):
    """Check exact decomposition via random sampling.
    C[i,k] = sum_r W[k*n+i, r] * (sum_{i',j'} U[i'*m+j', r] A[i',j']) * (sum_{j',k'} V[j'*p+k', r] B[j',k'])
    """
    rng = np.random.RandomState(seed)
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    for _ in range(n_tests):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        AB = A @ B
        C = np.zeros((n, p))
        for r in range(rank):
            a_part = np.dot(U[:, r], A)  # sum U[i*m+j] A[i,j]
            b_part = np.dot(V[:, r], B)  # sum V[j*p+k] B[j,k]
            w_r = W[:, r].reshape(p, n)  # w_r[k, i]
            C += w_r.T * (a_part * b_part)  # C[i,k] += w_r[k,i] * a_part * b_part
        if not np.allclose(C, AB, atol=1e-7, rtol=1e-7):
            return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    n, m, p = DIMENSIONS
    
    def build_structured():
        terms = []
        def add(u, v, w):
            terms.append((u, v, w))
        def Uv(*pairs):
            u = np.zeros(n*m)
            for (ii,jj,s) in pairs:
                u[ii*m+jj] += s
            return u
        def Vv(*pairs):
            v = np.zeros(m*p)
            for (jj,kk,s) in pairs:
                v[jj*p+kk] += s
            return v
        def Wv(*pairs):
            w = np.zeros(p*n)
            for (kk,ii,s) in pairs:
                w[kk*n+ii] += s
            return w
        def strassen(i0, j0, k0):
            # M1 = (A11+A22)(B11+B22) -> C11+1, C22+1
            add(Uv((i0,j0,1),(i0+1,j0+1,1)),
                Vv((j0,k0,1),(j0+1,k0+1,1)),
                Wv((k0,i0,1),(k0+1,i0+1,1)))
            # M2 = (A21+A22)B11 -> C21+1, C22-1
            add(Uv((i0+1,j0,1),(i0+1,j0+1,1)),
                Vv((j0,k0,1)),
                Wv((k0,i0+1,1),(k0+1,i0+1,-1)))
            # M3 = A11(B12-B22) -> C12+1, C22+1
            add(Uv((i0,j0,1)),
                Vv((j0,k0+1,1),(j0+1,k0+1,-1)),
                Wv((k0+1,i0,1),(k0+1,i0+1,1)))
            # M4 = A22(B21-B11) -> C11+1, C21+1
            add(Uv((i0+1,j0+1,1)),
                Vv((j0+1,k0,1),(j0,k0,-1)),
                Wv((k0,i0,1),(k0,i0+1,1)))
            # M5 = (A11+A12)B22 -> C11-1, C12+1
            add(Uv((i0,j0,1),(i0,j0+1,1)),
                Vv((j0+1,k0+1,1)),
                Wv((k0,i0,-1),(k0+1,i0,1)))
            # M6 = (A21-A11)(B11+B12) -> C22+1
            add(Uv((i0+1,j0,1),(i0,j0,-1)),
                Vv((j0,k0,1),(j0,k0+1,1)),
                Wv((k0+1,i0+1,1)))
            # M7 = (A12-A22)(B21+B22) -> C11+1
            add(Uv((i0,j0+1,1),(i0+1,j0+1,-1)),
                Vv((j0+1,k0,1),(j0+1,k0+1,1)),
                Wv((k0,i0,1)))
        def schoolbook(i0, j0, k0, ni, nj, nk):
            for di in range(ni):
                for dj in range(nj):
                    for dk in range(nk):
                        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
                        u[(i0+di)*m + (j0+dj)] = 1
                        v[(j0+dj)*p + (k0+dk)] = 1
                        w[(k0+dk)*n + (i0+di)] = 1
                        terms.append((u, v, w))
        
        # Strassen on (0,0,0) and (0,0,2)
        strassen(0, 0, 0)
        strassen(0, 0, 2)
        # Schoolbook rest
        schoolbook(0, 2, 0, 2, 1, 2)   # j=2, k=0,1
        schoolbook(0, 0, 4, 2, 2, 1)   # j=0,1, k=4
        schoolbook(0, 2, 2, 2, 1, 2)   # j=2, k=2,3
        schoolbook(0, 2, 4, 2, 1, 1)   # j=2, k=4
        
        U = np.column_stack([t[0] for t in terms])
        V = np.column_stack([t[1] for t in terms])
        W = np.column_stack([t[2] for t in terms])
        return U, V, W
    
    U, V, W = build_structured()
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    if not _validate(U, V, W, n, m, p, seed=42, n_tests=5):
        U, V, W = _schoolbook(n, m, p)
        best_rank = U.shape[1]
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Random rank reduction
    rng = np.random.RandomState(0)
    attempts = 0
    while time.time() - start_time < DEADLINE and attempts < 50000:
        attempts += 1
        rank = best_U.shape[1]
        if rank <= 24:
            break
        r = rng.randint(0, rank)
        U_c = np.delete(best_U, r, axis=1)
        V_c = np.delete(best_V, r, axis=1)
        W_c = np.delete(best_W, r, axis=1)
        if _validate(U_c, V_c, W_c, n, m, p, seed=attempts, n_tests=3):
            best_rank = rank - 1
            best_U, best_V, best_W = U_c.copy(), V_c.copy(), W_c.copy()
    
    def to_list(M):
        return [[float(x) for x in row] for row in M]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
