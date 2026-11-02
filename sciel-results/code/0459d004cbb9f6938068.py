import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def _build_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i_out in range(n):
        for j in range(m):
            for k_out in range(p):
                T[i_out*m + j, j*p + k_out, k_out*n + i_out] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-10):
    R = U.shape[1]
    Tc = np.zeros_like(T)
    for r in range(R):
        Tc += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
    return np.allclose(Tc, T, atol=tol)

def _strassen_terms_2x2():
    # Returns list of 7 tuples (u, v, w) for 2x2x2
    # Indices: 0:00, 1:01, 2:10, 3:11
    terms = []
    # M1
    u = {0: 1, 3: 1}
    v = {0: 1, 3: 1}
    w = {0: 1, 3: 1}
    terms.append((u, v, w))
    # M2
    u = {2: 1, 3: 1}
    v = {0: 1}
    w = {2: 1, 3: -1}
    terms.append((u, v, w))
    # M3
    u = {0: 1}
    v = {1: 1, 3: -1}
    w = {1: 1, 3: 1}
    terms.append((u, v, w))
    # M4
    u = {3: 1}
    v = {2: 1, 0: -1}
    w = {0: 1, 2: 1}
    terms.append((u, v, w))
    # M5
    u = {0: 1, 1: 1}
    v = {3: 1}
    w = {0: -1, 1: 1}
    terms.append((u, v, w))
    # M6
    u = {2: 1, 0: -1}
    v = {0: 1, 1: 1}
    w = {3: 1}
    terms.append((u, v, w))
    # M7
    u = {1: 1, 3: -1}
    v = {2: 1, 3: 1}
    w = {0: 1}
    terms.append((u, v, w))
    return terms

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    T = _build_tensor()
    
    # Build the rank-28 Strassen construction
    R = 28
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)

    S = _strassen_terms_2x2()

    def mA(li, off):
        r, c = li // 2, li % 2
        return r * m + (off * 2 + c)

    def mB(li, ro, co):
        r, c = li // 2, li % 2
        return (ro * 2 + r) * p + (co * 2 + c)

    def mC(li, co):
        r, c = li // 2, li % 2
        return (co * 2 + c) * n + r

    blocks = [
        (0, 0, 0, 0, 0),
        (1, 1, 0, 0, 7),
        (0, 0, 1, 1, 14),
        (1, 1, 1, 1, 21),
    ]

    for a_off, b_ro, b_co, c_co, base in blocks:
        if time.time() > deadline:
            break
        for j, (u_loc, v_loc, w_loc) in enumerate(S):
            ci = base + j
            for li, cf in u_loc.items():
                U[mA(li, a_off), ci] += cf
            for li, cf in v_loc.items():
                V[mB(li, b_ro, b_co), ci] += cf
            for li, cf in w_loc.items():
                W[mC(li, c_co), ci] += cf

    best_U, best_V, best_W = U, V, W
    best_rank = R

    # Verify the construction
    if not _verify(best_U, best_V, best_W, T):
        # Fallback to schoolbook rank 32
        R = 32
        U = np.zeros((n*m, R), dtype=float)
        V = np.zeros((m*p, R), dtype=float)
        W = np.zeros((p*n, R), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i*m*p + j*p + k
                    U[i*m+j, r] = 1
                    V[j*p+k, r] = 1
                    W[k*n+i, r] = 1
        best_U, best_V, best_W = U, V, W
        best_rank = R

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
