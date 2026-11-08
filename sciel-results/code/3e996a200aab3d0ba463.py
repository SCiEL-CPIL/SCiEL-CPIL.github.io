import numpy as np

def construct():
    n, m, p = 4, 4, 7
    terms = []

    def add_term(u, v, w):
        terms.append((u, v, w))

    def zero_u(): return [0] * (n * m)
    def zero_v(): return [0] * (m * p)
    def zero_w(): return [0] * (p * n)

    # 2x2 Strassen terms: (A_local, B_local, C_local)
    # Each is a list of (index, coeff)
    # Local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    s2_terms = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),  # M1
        ([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]),       # M2
        ([(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)]),       # M3
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),       # M4
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),       # M5
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),       # M6
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),       # M7
    ]

    def idx_A(i, j): return i * m + j
    def idx_B(i, j): return i * p + j
    def idx_C(i, j): return j * n + i  # W[k*n+i] where k=col, i=row

    # 4x4x4 Strassen: 7 outer * 7 inner = 49 terms
    # Outer Strassen on 2x2 blocks of 4x4
    # Block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # Each block is 2x2, local indices 0-3
    outer_terms = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),  # M1
        ([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]),       # M2
        ([(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)]),       # M3
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),       # M4
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),       # M5
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),       # M6
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),       # M7
    ]

    for oa, ob, oc in outer_terms:
        for ia, ib, ic in s2_terms:
            u = zero_u()
            v = zero_v()
            w = zero_w()

            # A part: outer block index -> 2x2 submatrix
            # Block b covers rows 2*(b//2)..2*(b//2)+1, cols 2*(b%2)..2*(b%2)+1
            for (ba, ca) in oa:
                br, bc = ba // 2, ba % 2
                for (la, da) in ia:
                    lr, lc = la // 2, la % 2
                    i = 2*br + lr
                    j = 2*bc + lc
                    u[idx_A(i, j)] += ca * da

            # B part: outer block index -> 2x2 submatrix of B
            # B is 4x7, blocks are 2x2 in the 4x4 part (cols 0-3)
            for (bb, cb) in ob:
                br, bc = bb // 2, bb % 2
                for (lb, db) in ib:
                    lr, lc = lb // 2, lb % 2
                    i = 2*br + lr
                    j = 2*bc + lc
                    v[idx_B(i, j)] += cb * db

            # C part: outer block index -> 2x2 submatrix of C
            # C is 4x4 for this part (cols 0-3)
            for (bc, cc) in oc:
                br, bc2 = bc // 2, bc % 2
                for (lc, dc) in ic:
                    lr, lc2 = lc // 2, lc % 2
                    i = 2*br + lr
                    j = 2*bc2 + lc2
                    w[idx_C(i, j)] += cc * dc

            add_term(u, v, w)

    # 4x4x2 for cols 4-5: 4 blocks of 2x2 * 2x2, each with 7 terms = 28 terms
    for i_b in range(2):
        for j_b in range(2):
            i0 = 2 * i_b
            j0 = 2 * j_b
            k0 = 4
            for ia, ib, ic in s2_terms:
                u = zero_u()
                v = zero_v()
                w = zero_w()

                for (la, da) in ia:
                    lr, lc = la // 2, la % 2
                    i = i0 + lr
                    j = j0 + lc
                    u[idx_A(i, j)] += da

                for (lb, db) in ib:
                    lr, lc = lb // 2, lb % 2
                    i = j0 + lr
                    j = k0 + lc
                    v[idx_B(i, j)] += db

                for (lc, dc) in ic:
                    lr, lc2 = lc // 2, lc % 2
                    i = i0 + lr
                    j = k0 + lc2
                    w[idx_C(i, j)] += dc

                add_term(u, v, w)

    # 4x4x1 for col 6: schoolbook = 16 terms
    for i in range(n):
        for j in range(m):
            u = zero_u()
            v = zero_v()
            w = zero_w()
            u[idx_A(i, j)] = 1
            v[idx_B(j, 6)] = 1
            w[idx_C(i, 6)] = 1
            add_term(u, v, w)

    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            if u[r] != 0:
                U[r][c] = u[r]
        for r in range(m * p):
            if v[r] != 0:
                V[r][c] = v[r]
        for r in range(p * n):
            if w[r] != 0:
                W[r][c] = w[r]

    return {"U": U, "V": V, "W": W}
