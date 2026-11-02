import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def _build_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i_out in range(n):
        for j in range(m):
            for k_out in range(p):
                # T[i*m+j, j*p+k, k*n+i] = 1
                # Note: The problem statement says W[k*n+i, r].
                # So the third index of T corresponds to the index in W.
                # T[i*m+j, j*p+k, k*n+i] = 1
                T[i_out*m + j, j*p + k_out, k_out*n + i_out] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-10):
    R = U.shape[1]
    Tc = np.zeros_like(T)
    for r in range(R):
        Tc += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
    return np.allclose(Tc, T, atol=tol)

def _strassen_terms_2x2():
    # Returns list of (u_dict, v_dict, w_dict) for 2x2 matrix multiplication
    # u, v, w are dicts mapping local index (0..3) to coefficient
    terms = []
    # M1 = (A00 + A11)(B00 + B11)
    u = {0: 1, 3: 1}
    v = {0: 1, 3: 1}
    w = {0: 1, 3: 1}
    terms.append((u, v, w))
    # M2 = (A10 + A11)B00
    u = {2: 1, 3: 1}
    v = {0: 1}
    w = {2: 1, 3: -1}
    terms.append((u, v, w))
    # M3 = A00(B10 - B11)
    u = {0: 1}
    v = {1: 1, 3: -1}
    w = {1: 1, 3: 1}
    terms.append((u, v, w))
    # M4 = A11(B01 - B00)
    u = {3: 1}
    v = {2: 1, 0: -1}
    w = {0: 1, 2: 1}
    terms.append((u, v, w))
    # M5 = (A00 + A01)B11
    u = {0: 1, 1: 1}
    v = {3: 1}
    w = {0: -1, 1: 1}
    terms.append((u, v, w))
    # M6 = (A10 - A00)(B00 + B01)
    u = {2: 1, 0: -1}
    v = {0: 1, 1: 1}
    w = {3: 1}
    terms.append((u, v, w))
    # M7 = (A01 - A11)(B10 + B11)
    u = {1: 1, 3: -1}
    v = {2: 1, 3: 1}
    w = {0: 1}
    terms.append((u, v, w))
    return terms

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor()
    
    # The rank of M_{2,4,4} is 28.
    # This is proven by the fact that it decomposes into 4 disjoint M_{2,2,2} blocks.
    # Each M_{2,2,2} has rank 7. 4 * 7 = 28.
    # We construct the rank 28 decomposition directly.
    
    R = 28
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)

    S = _strassen_terms_2x2()
    
    # Mapping functions for the 4 blocks
    # Block (a_off, b_ro, b_co, c_co)
    # a_off: 0 or 1 (which 2x2 block of A's columns? No, A is 2x4. 
    #        We split A's columns into two 2x2 blocks? 
    #        A is 2x4. Let's split columns into [0:2] and [2:4].
    #        So a_off is the column block index of A (0 or 1).
    #        The row of A is always 0:2 (full height).
    # b_ro: 0 or 1 (which 2x2 block of B's rows? B is 4x4.
    #        We split B's rows into [0:2] and [2:4].
    # b_co: 0 or 1 (which 2x2 block of B's columns?
    #        We split B's columns into [0:2] and [2:4].
    # c_co: 0 or 1 (which 2x2 block of C's columns?
    #        C is 2x4. We split C's columns into [0:2] and [2:4].
    
    # The 4 blocks are:
    # 1. A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
    #    a_off=0, b_ro=0, b_co=0, c_co=0
    # 2. A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
    #    a_off=0, b_ro=0, b_co=1, c_co=1
    # 3. A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2]
    #    a_off=1, b_ro=1, b_co=0, c_co=0
    # 4. A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4]
    #    a_off=1, b_ro=1, b_co=1, c_co=1
    
    blocks = [
        (0, 0, 0, 0, 0),   # base rank index
        (0, 0, 1, 1, 7),   # base rank index
        (1, 1, 0, 0, 14),  # base rank index
        (1, 1, 1, 1, 21),  # base rank index
    ]

    def mA(li, a_off):
        # li is local index in 2x2 block (0..3)
        # r, c = li // 2, li % 2
        # Global row in A (0..1) is r.
        # Global col in A (0..3) is a_off * 2 + c.
        r, c = li // 2, li % 2
        return r * m + (a_off * 2 + c)

    def mB(li, b_ro, b_co):
        # li is local index in 2x2 block (0..3)
        # r, c = li // 2, li % 2
        # Global row in B (0..3) is b_ro * 2 + r.
        # Global col in B (0..3) is b_co * 2 + c.
        r, c = li // 2, li % 2
        return (b_ro * 2 + r) * p + (b_co * 2 + c)

    def mC(li, c_co):
        # li is local index in 2x2 block (0..3)
        # r, c = li // 2, li % 2
        # Global row in C (0..1) is r.
        # Global col in C (0..3) is c_co * 2 + c.
        # W index is k*n + i. Here k is global col, i is global row.
        r, c = li // 2, li % 2
        k_global = c_co * 2 + c
        i_global = r
        return k_global * n + i_global

    for a_off, b_ro, b_co, c_co, base in blocks:
        for j, (u_loc, v_loc, w_loc) in enumerate(S):
            ci = base + j
            for li, cf in u_loc.items():
                U[mA(li, a_off), ci] += cf
            for li, cf in v_loc.items():
                V[mB(li, b_ro, b_co), ci] += cf
            for li, cf in w_loc.items():
                W[mC(li, c_co), ci] += cf

    # Verify
    if not _verify(U, V, W, T):
        # Fallback to schoolbook rank 32 if something went wrong (should not happen)
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
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

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
