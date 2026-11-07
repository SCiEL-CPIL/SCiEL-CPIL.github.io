"""Clique partition of split graphs: per-vertex greedy clique packing + C-edge 2-clique fill."""
import time
from collections import defaultdict

_DEADLINE = time.time() + 140.0


def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _solve_one(n, k, cross):
        # cross: set of (a, v) with a < k <= v
        # N_C[v] = neighbors of v in C
        N_C = {v: set() for v in range(k, n)}
        for (a, v) in cross:
            N_C[v].add(a)

        # free C-internal edges (each starts as its own future 2-clique unless merged)
        free_cint = set()
        for a in range(k):
            for b in range(a + 1, k):
                free_cint.add((a, b))

        cliques = []
        for v in range(k, n):
            remaining = set(N_C[v])
            while remaining:
                if time.time() > _DEADLINE:
                    break
                a0 = min(remaining)
                S = {a0}
                changed = True
                while changed:
                    changed = False
                    for b in list(remaining):
                        if b in S:
                            continue
                        ok = True
                        for a in S:
                            mn, mx = (a, b) if a < b else (b, a)
                            if (mn, mx) not in free_cint:
                                ok = False
                                break
                        if ok:
                            S.add(b)
                            remaining.discard(b)
                            changed = True
                Slist = sorted(S)
                for i in range(len(Slist)):
                    for j in range(i + 1, len(Slist)):
                        free_cint.discard((Slist[i], Slist[j]))
                cliques.append(Slist + [v])

        # fill remaining free C-int edges with 2-cliques
        for (a, b) in free_cint:
            cliques.append([a, b])
        return cliques

    def _safe_fallback(n, k, cross):
        cliques = [list(range(k))]
        for v in range(k, n):
            for a in range(k):
                if (a, v) in cross:
                    cliques.append([a, v])
        return cliques

    def _verify(cliques, edges_set):
        ec = defaultdict(int)
        for cl in cliques:
            s = sorted(cl)
            for i in range(len(s)):
                for j in range(i + 1, len(s)):
                    ec[(s[i], s[j])] += 1
        for e in edges_set:
            if ec.get(e, 0) != 1:
                return False
        for e, c in ec.items():
            if c != 1:
                return False
        return True

    results = []
    for (n, k, p, seed) in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        edges = instance(n, k, p, seed)
        edges_set = set(edges)
        cross = set()
        for (u, v) in edges:
            if u < k and v >= k:
                cross.add((u, v))
            elif v < k and u >= k:
                cross.add((v, u))

        if time.time() > _DEADLINE:
            cliques = _safe_fallback(n, k, cross)
        else:
            cliques = _solve_one(n, k, cross)
            if not _verify(cliques, edges_set):
                cliques = _safe_fallback(n, k, cross)

        cliques = [list(map(int, cl)) for cl in cliques]
        results.append(cliques)

    return results
