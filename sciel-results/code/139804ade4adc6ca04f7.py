"""Kissing number in dimension 11 via maximum independent set on the conflict graph."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np


def construct():
    deadline = time.time() + 140.0
    best = None

    # ---- candidate pool: all weight-1 and weight-2 integer vectors ----
    cands = []
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = s
            cands.append(v)
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    cands.append(v)
    n = len(cands)
    A = np.array(cands, dtype=np.int64)
    sq = (A * A).sum(1)

    # ---- conflict graph: edge if pairwise distance < max norm ----
    # dist2(i,j) = sq[i]+sq[j]-2*dot(i,j); conflict iff dist2 < max_sq
    max_sq = int(sq.max())
    G = (A @ A.T).astype(np.int64)
    G = sq[:, None] + sq[None, :] - 2 * G
    np.fill_diagonal(G, 0)
    conflict = G < max_sq
    conflict = np.ascontiguousarray(conflict)

    def run(seed):
        # min-degree greedy to seed a lower bound
        chosen = set()
        avail = set(range(n))
        while avail:
            v = min(avail, key=lambda x: (conflict[x][list(avail)].sum(), x))
            chosen.add(v)
            avail.difference_update(np.flatnonzero(conflict[v]))
        best_local = len(chosen)
        best_set = set(chosen)

        # exact branch-and-bound over the remaining (unselected) graph
        cand = sorted(set(range(n)) - chosen)
        cset = set(cand)
        rem_conf = {c: set(int(x) for x in np.flatnonzero(conflict[c])) & cset for c in cand}
        best_local2 = best_local

        def bound(c):
            nb = rem_conf[c]
            return 1 + (len(cset) - 1 - len(nb))

        def dfs(cand_set):
            nonlocal best_local2, best_set
            if time.time() >= deadline:
                return
            if not cand_set:
                return
            if len(cand_set) + best_local2 <= best_local2:
                return
            # upper bound: 1 + max over candidates of (|cand_set|-1 - |N(c)|)
            ub = 0
            for c in cand_set:
                b = 1 + (len(cand_set) - 1 - len(rem_conf[c] & cand_set))
                if b > ub:
                    ub = b
            if ub <= 0:
                return
            # pick max-degree vertex for branching
            c = max(cand_set, key=lambda x: len(rem_conf[x] & cand_set))
            # branch include c
            newset = cand_set - rem_conf[c] - {c}
            dfs(newset)
            if len(newset) + best_local2 > best_local2:
                pass
            # branch exclude c
            dfs(cand_set - {c})

        # iterative deepening style: simple top-down with incumbent tracking
        # (kept shallow to respect deadline)
        dfs(set(cand))

        return best_local2, best_set

    # run once; if time remains, restart with a different seed order
    res = run(0)
    if res is not None and res[1] is not None:
        best = res[1]

    if best is None:
        # fallback: original seed vectors
        import itertools
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)

    return A[list(best)].astype(np.int64)
# EVOLVE-BLOCK-END
