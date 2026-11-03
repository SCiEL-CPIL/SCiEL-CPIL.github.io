import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 4, 4, 6
    
    # Outer Strassen (Rank 7)
    # Inner Rank 8 (Hypothesis)
    rank = 56

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    outer_steps = [
        (([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)])),
        (([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)])),
        (([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)])),
        (([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)])),
        (([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (0, -1)])),
        (([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)])),
        (([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])),
    ]

    def gA(b, r, s):
        return (b // 2) * 2 + r, (b % 2) * 2 + s

    def gB(b, s, k):
        return (b // 2) * 2 + s, (b % 2) * 3 + k

    def gC(b, r, k):
        return (b // 2) * 2 + r, (b % 2) * 3 + k

    # Inner Rank 8 Decomposition for 2x2 x 2x3
    # Terms 0-6: Strassen for cols 0,1
    # Term 7: Handling col 2
    inner_decomp = [
        # 0
        (([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)])),
        # 1
        (([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)], [(1, 0, 1), (1, 1, -1)])),
        # 2
        (([(0, 0, 1)], [(0, 1, 1), (1, 1, -1)], [(0, 1, 1), (1, 1, 1)])),
        # 3
        (([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (1, 0, 1)])),
        # 4
        (([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)], [(0, 1, 1), (0, 0, -1)])),
        # 5
        (([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)], [(1, 1, 1)])),
        # 6
        (([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)], [(0, 0, 1)])),
        # 7: Proposed term for col 2
        # M7 = (a00 + a10 + a01 + a11) * (b02 + b12)
        # This is a guess. If it fails, verification will catch it.
        (([(0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)], [(0, 2, 1), (1, 2, 1)], [(0, 2, 1), (1, 2, 1)])),
    ]

    def add_term(rank_idx, A_terms, B_terms, C_terms, A_outer, B_outer, C_outer):
        for (r, s, ca) in A_terms:
            for (b, cb) in A_outer:
                I, J = gA(b, r, s)
                U[I * m + J][rank_idx] += ca * cb
        for (s, k, cb) in B_terms:
            for (b, cb2) in B_outer:
                J, K = gB(b, s, k)
                V[J * p + K][rank_idx] += cb * cb2
        for (r, k, cc) in C_terms:
            for (b, cb2) in C_outer:
                I, K = gC(b, r, k)
                W[K * n + I][rank_idx] += cc * cb2

    for i, (A_outer, B_outer, C_outer) in enumerate(outer_steps):
        base = i * 8
        for j, (Ai, Bi, Ci) in enumerate(inner_decomp):
            add_term(base + j, Ai, Bi, Ci, A_outer, B_outer, C_outer)

    # Verification
    if time.time() < deadline - 5:
        Um = np.array(U)
        Vm = np.array(V)
        Wm = np.array(W)
        T = np.zeros((n, m, p))
        for r in range(rank):
            u = Um[:, r]
            v = Vm[:, r]
            w = Wm[:, r]
            for i in range(n):
                for j in range(m):
                    if u[i*m+j] == 0:
                        continue
                    for k in range(p):
                        T[i, j, k] += u[i*m+j] * v[j*p+k] * w[k*n+i]
        ok = np.allclose(T, 1.0)
        if not ok:
            # Fallback to Rank 77
            rank2 = 77
            U2 = [[0] * rank2 for _ in range(n * m)]
            V2 = [[0] * rank2 for _ in range(m * p)]
            W2 = [[0] * rank2 for _ in range(p * n)]
            
            inner_strassen = [
                (([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)])),
                (([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)], [(1, 0, 1), (1, 1, -1)])),
                (([(0, 0, 1)], [(0, 1, 1), (1, 1, -1)], [(0, 1, 1), (1, 1, 1)])),
                (([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (1, 0, 1)])),
                (([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)], [(0, 1, 1), (0, 0, -1)])),
                (([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)], [(1, 1, 1)])),
                (([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)], [(0, 0, 1)])),
            ]
            inner_schoolbook = [
                (([(0, 0, 1)], [(0, 2, 1)], [(0, 2, 1)])),
                (([(0, 1, 1)], [(1, 2, 1)], [(0, 2, 1)])),
                (([(1, 0, 1)], [(0, 2, 1)], [(1, 2, 1)])),
                (([(1, 1, 1)], [(1, 2, 1)], [(1, 2, 1)])),
            ]
            
            for i, (A_outer, B_outer, C_outer) in enumerate(outer_steps):
                base = i * 11
                for j, (Ai, Bi, Ci) in enumerate(inner_strassen):
                    add_term(base + j, Ai, Bi, Ci, A_outer, B_outer, C_outer)
                for j, (Ai, Bi, Ci) in enumerate(inner_schoolbook):
                    add_term(base + 7 + j, Ai, Bi, Ci, A_outer, B_outer, C_outer)
            
            return {"U": U2, "V": V2, "W": W2}

    return {"U": U, "V": V, "W": W}
