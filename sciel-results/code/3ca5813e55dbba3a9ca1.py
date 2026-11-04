import time
_t0 = time.time()
_LIMIT = 150.0
_PAD = 5.0

def construct():
    n, m, p = 4, 4, 9
    terms = []

    def emit(u, v, w):
        terms.append((u, v, w))

    # 7 rank-1 tensors for 2x2x2 Strassen
    S2 = [
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}),
        ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),
        ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
    ]

    def strassen_4x4x4(a_i0, a_j0, b_j0, b_k0, c_i0, c_k0):
        combos = [
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
            ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
            ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
            ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
            ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
            ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
        ]
        for ab, bb, cb in combos:
            for (u_loc, v_loc, w_loc) in S2:
                u = [0.0] * (n * m)
                v = [0.0] * (m * p)
                w = [0.0] * (p * n)
                for (li, lj), cf in u_loc.items():
                    for (br, bc, bcf) in ab:
                        gi = a_i0 + br * 2 + li
                        gj = a_j0 + bc * 2 + lj
                        u[gi * m + gj] += cf * bcf
                for (lj, lk), cf in v_loc.items():
                    for (br, bc, bcf) in bb:
                        gj = b_j0 + br * 2 + lj
                        gk = b_k0 + bc * 2 + lk
                        v[gj * p + gk] += cf * bcf
                for (li, lk), cf in w_loc.items():
                    for (br, bc, bcf) in cb:
                        gi = c_i0 + br * 2 + li
                        gk = c_k0 + bc * 2 + lk
                        w[gk * n + gi] += cf * bcf
                emit(u, v, w)

    strassen_4x4x4(0, 0, 0, 0, 0, 0)
    strassen_4x4x4(0, 0, 0, 4, 0, 4)

    for i in range(n):
        for j in range(m):
            u = [0.0] * (n * m)
            v = [0.0] * (m * p)
            w = [0.0] * (p * n)
            u[i * m + j] = 1.0
            v[j * p + 8] = 1.0
            w[8 * n + i] = 1.0
            emit(u, v, w)

    r = len(terms)
    elapsed = time.time() - _t0
    if elapsed > _LIMIT - _PAD:
        pass

    U = [[terms[k][0][i] for k in range(r)] for i in range(n * m)]
    V = [[terms[k][1][i] for k in range(r)] for i in range(m * p)]
    W = [[terms[k][2][i] for k in range(r)] for i in range(p * n)]

    return {"U": U, "V": V, "W": W}
