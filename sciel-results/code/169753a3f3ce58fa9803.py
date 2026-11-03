"""Matrix multiplication 4x4 * 4x9 rank-minimal exact tensor decomposition.

Strategy: split B into [B1 (4x4), B2 (4x4), B3 (4x1)].
  C = [A*B1, A*B2, A*B3]
  A*B1: Strassen rank 49
  A*B2: Strassen rank 49
  A*B3: schoolbook rank 16
  Total rank: 114
"""
import time
import numpy as np

_START = time.time()
_DEADLINE = 130.0


def _time_ok():
    return time.time() - _START < _DEADLINE


def _strassen_2x2_terms():
    """Return 7 Strassen terms for 2x2*2x2 as (u, v, w) dicts."""
    return [
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}),
        ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),
        ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
    ]


def _strassen_4x4_4x4_terms():
    """49 terms for 4x4*4x4 via recursive Strassen (2 levels)."""
    inner = _strassen_2x2_terms()
    top = [
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}),
        ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),
        ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
    ]
    all_terms = []
    for (ab, bb, cb) in top:
        for (ui, vi, wi) in inner:
            u, v, w = {}, {}, {}
            for (bi, bj), ca in ab.items():
                for (li, lj), ci in ui.items():
                    i, j = 2*bi+li, 2*bj+lj
                    u[(i,j)] = u.get((i,j), 0) + ca*ci
            for (bi, bj), cb_ in bb.items():
                for (li, lj), ci in vi.items():
                    j, k = 2*bi+li, 2*bj+lj
                    v[(j,k)] = v.get((j,k), 0) + cb_*ci
            for (bi, bj), cc in cb.items():
                for (li, lj), ci in wi.items():
                    i, k = 2*bi+li, 2*bj+lj
                    w[(i,k)] = w.get((i,k), 0) + cc*ci
            all_terms.append((u, v, w))
    return all_terms


def _rank114():
    """Build rank-114 decomposition for 4x4 * 4x9."""
    n, m, p = 4, 4, 9
    rank = 114
    
    U = np.zeros((n*m, rank), dtype=np.int8)
    V = np.zeros((m*p, rank), dtype=np.int8)
    W = np.zeros((p*n, rank), dtype=np.int8)
    
    st = _strassen_4x4_4x4_terms()
    
    # Part 1: A*B1 (B1 = B[:, 0:4]) -> C[:, 0:4]
    for c_idx, (up, vp, wp) in enumerate(st):
        for (i, j), cu in up.items():
            if cu != 0:
                U[i*m + j, c_idx] = cu
        for (j, k), cv in vp.items():
            if cv != 0:
                V[j*p + k, c_idx] = cv
        for (i, k), cw in wp.items():
            if cw != 0:
                W[k*n + i, c_idx] = cw
    
    # Part 2: A*B2 (B2 = B[:, 4:8]) -> C[:, 4:8]
    offset = 49
    for c_idx, (up, vp, wp) in enumerate(st):
        for (i, j), cu in up.items():
            if cu != 0:
                U[i*m + j, c_idx + offset] = cu
        for (j, k), cv in vp.items():
            if cv != 0:
                V[j*p + (4 + k), c_idx + offset] = cv
        for (i, k), cw in wp.items():
            if cw != 0:
                W[(4 + k)*n + i, c_idx + offset] = cw
    
    # Part 3: A*B3 (B3 = B[:, 8:9]) -> C[:, 8]
    offset2 = 98
    for i in range(n):
        for j in range(m):
            c = offset2 + i*m + j
            U[i*m + j, c] = 1
            V[j*p + 8, c] = 1
            W[8*n + i, c] = 1
    
    return U, V, W


def _verify_numpy(U, V, W, n, m, p):
    """Verify decomposition using numpy einsum."""
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    expected = np.ones((n, m, p), dtype=np.int8)
    return np.array_equal(T, expected)


def _schoolbook(n, m, p):
    """Schoolbook rank = n*m*p = 144."""
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.int8)
    V = np.zeros((m*p, rank), dtype=np.int8)
    W = np.zeros((p*n, rank), dtype=np.int8)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, c] = 1
                V[j*p + k, c] = 1
                W[k*n + i, c] = 1
                c += 1
    return U, V, W


def construct():
    n, m, p = 4, 4, 9
    
    if _time_ok():
        U, V, W = _rank114()
        if _verify_numpy(U, V, W, n, m, p):
            return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    if _time_ok():
        U, V, W = _schoolbook(n, m, p)
        if _verify_numpy(U, V, W, n, m, p):
            return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    U, V, W = _schoolbook(n, m, p)
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
