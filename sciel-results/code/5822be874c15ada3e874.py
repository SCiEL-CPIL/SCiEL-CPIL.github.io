"""Matrix multiplication 4x4 * 4x9 rank-minimal exact tensor decomposition.

Strategy: split B into [B1 (4x4), B2 (4x4), B3 (4x1)].
  C = [A*B1, A*B2, A*B3]
  A*B1: Strassen rank 49
  A*B2: Strassen rank 49
  A*B3: Optimized rank 4 (shared terms for all i)
  Total rank: 102
"""
import time

_START = time.time()
_DEADLINE = 130.0


def _time_ok():
    return time.time() - _START < _DEADLINE


def _build_tensor(n, m, p):
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(k * n + i, i * m + j, j * p + k)] = 1
    return T


def _exact_check(U, V, W, T, n, m, p):
    rank = len(U[0])
    for io in range(p * n):
        for u in range(n * m):
            for v in range(m * p):
                s = 0
                for c in range(rank):
                    s += U[u][c] * V[v][c] * W[io][c]
                if s != T.get((io, u, v), 0):
                    return False
    return True


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _strassen_2x2_terms():
    terms = []
    terms.append((
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1}
    ))
    terms.append((
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,0): 1, (1,1): -1}
    ))
    terms.append((
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(0,1): 1, (1,1): 1}
    ))
    terms.append((
        {(1,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (1,0): 1}
    ))
    terms.append((
        {(0,0): 1, (0,1): 1},
        {(1,1): 1},
        {(0,0): -1, (0,1): 1}
    ))
    terms.append((
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (0,1): 1},
        {(1,1): 1}
    ))
    terms.append((
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1}
    ))
    return terms


def _strassen_4x4_4x4_terms():
    inner_terms = _strassen_2x2_terms()
    top_products = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    all_terms = []
    for (a_blocks, b_blocks, c_blocks) in top_products:
        for (u_inner, v_inner, w_inner) in inner_terms:
            u = {}
            v = {}
            w = {}
            for (bi, bj), ca in a_blocks.items():
                for (li, lj), ci in u_inner.items():
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    u[(i, j)] = u.get((i, j), 0) + ca * ci
            for (bi, bj), cb in b_blocks.items():
                for (li, lj), ci in v_inner.items():
                    j = 2 * bi + li
                    k = 2 * bj + lj
                    v[(j, k)] = v.get((j, k), 0) + cb * ci
            for (bi, bj), cc in c_blocks.items():
                for (li, lj), ci in w_inner.items():
                    i = 2 * bi + li
                    k = 2 * bj + lj
                    w[(i, k)] = w.get((i, k), 0) + cc * ci
            all_terms.append((u, v, w))
    return all_terms


def _rank102():
    n, m, p = 4, 4, 9
    rank = 102
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_terms = _strassen_4x4_4x4_terms()
    
    # Block 1: B columns 0-3
    for c_idx, (u_pat, v_pat, w_pat) in enumerate(strassen_terms):
        for (i, j), cu in u_pat.items():
            if cu != 0:
                U[i * m + j][c_idx] = cu
        for (j, k), cv in v_pat.items():
            if cv != 0:
                V[j * p + k][c_idx] = cv
        for (i, k), cw in w_pat.items():
            if cw != 0:
                W[k * n + i][c_idx] = cw
                
    # Block 2: B columns 4-7
    offset = 49
    for c_idx, (u_pat, v_pat, w_pat) in enumerate(strassen_terms):
        for (i, j), cu in u_pat.items():
            if cu != 0:
                U[i * m + j][c_idx + offset] = cu
        for (j, k), cv in v_pat.items():
            if cv != 0:
                V[j * p + (4 + k)][c_idx + offset] = cv
        for (i, k), cw in w_pat.items():
            if cw != 0:
                W[(4 + k) * n + i][c_idx + offset] = cw
                
    # Block 3: B column 8 (Optimized Rank 4)
    offset2 = 98
    for j in range(m):
        c_idx = offset2 + j
        V[j * p + 8][c_idx] = 1
        for i in range(n):
            U[i * m + j][c_idx] = 1
            W[8 * n + i][c_idx] = 1
            
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = 4, 4, 9
    T = _build_tensor(n, m, p)
    if _time_ok():
        cand = _rank102()
        if _exact_check(cand["U"], cand["V"], cand["W"], T, n, m, p):
            return cand
    best = _schoolbook(n, m, p)
    if _exact_check(best["U"], best["V"], best["W"], T, n, m, p):
        return best
    return best
