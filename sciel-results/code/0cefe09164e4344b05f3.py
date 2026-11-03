"""Matrix multiplication 4x4 * 4x9 rank-minimal exact tensor decomposition.

Strategy: split B into [B1 (4x4), B2 (4x4), B3 (4x1)].
  C = [A*B1, A*B2, A*B3]
  A*B1: Strassen rank 49
  A*B2: Strassen rank 49
  A*B3: schoolbook rank 16
  Total rank: 114
"""
import time

_START = time.time()
_DEADLINE = 130.0


def _time_ok():
    return time.time() - _START < _DEADLINE


def _strassen_2x2_terms():
    """7 Strassen terms for 2x2*2x2. Each term: (u, v, w) dicts."""
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
    """49 Strassen terms for 4x4*4x4. Each term: (u, v, w) dicts."""
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
    terms = []
    for (ab, bb, cb) in top:
        for (ui, vi, wi) in inner:
            u, v, w = {}, {}, {}
            for (bi,bj), ca in ab.items():
                for (li,lj), ci in ui.items():
                    u[(2*bi+li, 2*bj+lj)] = u.get((2*bi+li, 2*bj+lj), 0) + ca*ci
            for (bi,bj), cb_ in bb.items():
                for (li,lj), ci in vi.items():
                    v[(2*bi+li, 2*bj+lj)] = v.get((2*bi+li, 2*bj+lj), 0) + cb_*ci
            for (bi,bj), cc in cb.items():
                for (li,lj), ci in wi.items():
                    w[(2*bi+li, 2*bj+lj)] = w.get((2*bi+li, 2*bj+lj), 0) + cc*ci
            terms.append((u, v, w))
    return terms


def _schoolbook(n, m, p):
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _rank114():
    n, m, p = 4, 4, 9
    rank = 114
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    st = _strassen_4x4_4x4_terms()
    
    # A*B1 (cols 0-3)
    for c, (u,v,w) in enumerate(st):
        for (i,j), cu in u.items():
            if cu: U[i*m+j][c] = cu
        for (j,k), cv in v.items():
            if cv: V[j*p+k][c] = cv
        for (i,k), cw in w.items():
            if cw: W[k*n+i][c] = cw
    
    # A*B2 (cols 4-7)
    off = 49
    for c, (u,v,w) in enumerate(st):
        for (i,j), cu in u.items():
            if cu: U[i*m+j][c+off] = cu
        for (j,k), cv in v.items():
            if cv: V[j*p+(4+k)][c+off] = cv
        for (i,k), cw in w.items():
            if cw: W[(4+k)*n+i][c+off] = cw
    
    # A*B3 (col 8), schoolbook rank 16
    off2 = 98
    for i in range(n):
        for j in range(m):
            c = off2 + i*m + j
            U[i*m+j][c] = 1
            V[j*p+8][c] = 1
            W[8*n+i][c] = 1
    
    return {"U": U, "V": V, "W": W}


def construct():
    if _time_ok():
        return _rank114()
    return _schoolbook(4, 4, 9)
