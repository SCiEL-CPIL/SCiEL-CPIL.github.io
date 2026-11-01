"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

import time
import numpy as np
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Target triples (a,b,c) that must evaluate to 1
    target = [(i * m + j, j * p + k, k * n + i)
              for i in range(n) for j in range(m) for k in range(p)]
    ta = np.array([t[0] for t in target], dtype=np.int64)
    tb = np.array([t[1] for t in target], dtype=np.int64)
    tc = np.array([t[2] for t in target], dtype=np.int64)

    def make_schoolbook():
        R = n * m * p
        U = np.zeros((nm, R), dtype=int)
        V = np.zeros((mp, R), dtype=int)
        W = np.zeros((pn, R), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def verify(U, V, W, R):
        # (a,b,c) -> sum_r U[a,r]*V[b,r]*W[c,r] must equal 1 for all target triples
        vals = (U[ta, :R] * V[tb, :R] * W[tc, :R]).sum(axis=1)
        return bool(np.all(vals == 1))

    def try_remove(U, V, W, R, c):
        """Try removing column c. Returns (U', V', W', R', success)."""
        idx = np.arange(R)
        mask = idx != c
        Ut = U[:, mask].copy()
        Vt = V[:, mask].copy()
        Wt = W[:, mask].copy()
        Rt = R - 1
        ok = verify(Ut, Vt, Wt, Rt)
        return Ut, Vt, Wt, Rt, ok

    def try_merge(U, V, W, R, c1, c2):
        """Merge columns c1 and c2 (same U,V support). Returns (U', V', W', R', success)."""
        # After merging, U and V lose one column; W column c1 gets W[c1,c1]+W[c1,c2]
        idx = np.arange(R)
        mask = idx != c2
        Ut = U[:, mask].copy()
        Vt = V[:, mask].copy()
        Wt = W[:, mask].copy()
        Rt = R - 1
        # Adjust W: the merged column is at position c1 if c1 < c2, else c1-1
        pos1 = c1 if c1 < c2 else c1 - 1
        Wt[:, pos1] = W[:, c1] + W[:, c2]
        ok = verify(Ut, Vt, Wt, Rt)
        return Ut, Vt, Wt, Rt, ok

    def get_supports(U, V, R):
        """Return dict mapping column index to (U_support_tuple, V_support_tuple)."""
        sup = {}
        for c in range(R):
            us = tuple(np.nonzero(U[:, c])[0])
            vs = tuple(np.nonzero(V[:, c])[0])
            sup[c] = (us, vs)
        return sup

    def group_by_support(U, V, R):
        """Group columns by their (U_support, V_support)."""
        sup = get_supports(U, V, R)
        groups = {}
        for c in range(R):
            key = sup[c]
            groups.setdefault(key, []).append(c)
        return groups

    U, V, W = make_schoolbook()
    R = U.shape[1]
    best_U, best_V, best_W, best_R = U.copy(), V.copy(), W.copy(), R

    def search_round(U, V, W, R, deadline, seed):
        """Run a greedy removal + merge search. Returns (U, V, W, R)."""
        rng = random.Random(seed)
        improved = True
        while improved and (time.time() < deadline):
            improved = False
            # Phase 1: try removals in random order
            cols = list(range(R))
            rng.shuffle(cols)
            for c in cols:
                if time.time() > deadline:
                    break
                Ut, Vt, Wt, Rt, ok = try_remove(U, V, W, R, c)
                if ok:
                    U, V, W, R = Ut, Vt, Wt, Rt
                    improved = True
                    break

            # Phase 2: try merges
            if not improved and R > 1:
                groups = group_by_support(U, V, R)
                for key, members in groups.items():
                    if time.time() > deadline:
                        break
                    if len(members) < 2:
                        continue
                    # Try all pairs in this group
                    for i in range(len(members)):
                        for j in range(i + 1, len(members)):
                            if time.time() > deadline:
                                break
                            c1, c2 = members[i], members[j]
                            Ut, Vt, Wt, Rt, ok = try_merge(U, V, W, R, c1, c2)
                            if ok:
                                U, V, W, R = Ut, Vt, Wt, Rt
                                improved = True
                                break
                        if improved:
                            break
        return U, V, W, R

    # Initial greedy search
    U, V, W, R = search_round(U, V, W, R, deadline, 12345)
    if R < best_R:
        best_U, best_V, best_W, best_R = U.copy(), V.copy(), W.copy(), R

    # Multiple randomized restarts from the current best
    seeds = [42, 1337, 999, 777, 555, 333, 222, 111, 888, 666]
    for seed in seeds:
        if time.time() > deadline:
            break
        U2, V2, W2, R2 = U.copy(), V.copy(), W.copy(), R
        U2, V2, W2, R2 = search_round(U2, V2, W2, R2, deadline, seed)
        if R2 < best_R:
            best_U, best_V, best_W, best_R = U2.copy(), V2.copy(), W2.copy(), R2
            U, V, W, R = U2, V2, W2, R2

    # Also try starting fresh from schoolbook with different seeds
    for seed in [1000, 2000, 3000, 4000, 5000]:
        if time.time() > deadline:
            break
        U3, V3, W3 = make_schoolbook()
        R3 = U3.shape[1]
        U3, V3, W3, R3 = search_round(U3, V3, W3, R3, deadline, seed)
        if R3 < best_R:
            best_U, best_V, best_W, best_R = U3.copy(), V3.copy(), W3.copy(), R3

    # Convert to lists of lists (integer entries)
    def to_list(A):
        return A.tolist()

    return {"U": to_list(best_U[:, :best_R]),
            "V": to_list(best_V[:, :best_R]),
            "W": to_list(best_W[:, :best_R])}
# EVOLVE-BLOCK-END
