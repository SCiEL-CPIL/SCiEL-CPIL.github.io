import random
from itertools import combinations

def _popcount(x):
    return bin(x).count('1')

def _try_build(n_target, max_T_size=7, n_random=20, cd_passes=2):
    adj = [0] * n_target
    for k in range(1, n_target):
        verts = list(range(k))
        # Precompute potentially-bad T's (subsets of 0..k-1 with 5 <= |T| <= max_T_size)
        P = []
        for size in range(5, min(max_T_size, k) + 1):
            for T in combinations(verts, size):
                Tmask = 0
                for v in T:
                    Tmask |= (1 << v)
                degs = {}
                for v in T:
                    degs[v] = _popcount(adj[v] & Tmask)
                forbidden = set()
                for d in range(0, size + 1):
                    count_dminus = 0
                    for v in T:
                        if degs[v] == d - 1:
                            count_dminus += 1
                    if count_dminus != d:
                        continue
                    ok = True
                    for v in T:
                        if degs[v] == d - 1:
                            continue
                        if degs[v] != d:
                            ok = False
                            break
                    if ok:
                        fmask = 0
                        for v in T:
                            if degs[v] == d - 1:
                                fmask |= (1 << v)
                        forbidden.add(fmask)
                if forbidden:
                    P.append((Tmask, forbidden))

        def cost(Smask):
            c = 0
            for Tmask, forbidden in P:
                if (Smask & Tmask) in forbidden:
                    c += 1
            return c

        best_S = 0
        best_cost = float('inf')
        full = 0
        for v in verts:
            full |= (1 << v)
        candidates = [0, full]
        for _ in range(n_random):
            S = 0
            for v in verts:
                if random.random() < 0.5:
                    S |= (1 << v)
            candidates.append(S)
        for S in candidates:
            c = cost(S)
            if c < best_cost:
                best_cost = c
                best_S = S

        improved = True
        passes = 0
        while improved and passes < cd_passes:
            improved = False
            passes += 1
            for v in verts:
                cur = (best_S >> v) & 1
                for add in (True, False):
                    if cur == (1 if add else 0):
                        continue
                    if add:
                        S2 = best_S | (1 << v)
                    else:
                        S2 = best_S & ~(1 << v)
                    c = cost(S2)
                    if c < best_cost:
                        best_cost = c
                        best_S = S2
                        improved = True

        for v in range(k):
            if (best_S >> v) & 1:
                adj[k] |= (1 << v)
                adj[v] |= (1 << k)

    # Verify: no regular induced subgraph of size >= 6
    n = n_target
    for size in range(6, n + 1):
        for T in combinations(range(n), size):
            Tmask = 0
            for v in T:
                Tmask |= (1 << v)
            first = -1
            reg = True
            for v in T:
                dd = _popcount(adj[v] & Tmask)
                if first == -1:
                    first = dd
                elif dd != first:
                    reg = False
                    break
            if reg:
                return None
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
    return {'n': n, 'edges': edges}

def construct():
    seeds = [42, 123, 7, 999, 555, 2024, 314, 271]
    for target in (21, 20):
        for seed in seeds:
            random.seed(seed)
            res = _try_build(target)
            if res is not None:
                return res
    return {'n': 5, 'edges': []}
