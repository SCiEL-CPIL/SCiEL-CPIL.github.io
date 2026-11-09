"""Clique partitions of three fixed split (chordal) graphs."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random

    t0 = time.time()
    DEADLINE = t0 + 110.0  # safety margin before 120s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def partition_cliques(n, k, p, seed):
        edges = instance(n, k, p, seed)
        cliques = []
        # 1 clique covering all clique-clique edges
        cliques.append(list(range(k)))
        # 1 clique per cross edge (mixed clique of size 2)
        for (u, v) in edges:
            if u >= k and v >= k:
                continue
            if u >= k:
                u, v = v, u
            if u < k and v >= k:
                cliques.append([u, v])
        return cliques

    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > DEADLINE:
            break
        cliques = partition_cliques(*spec)
        # Ensure every clique is a list of distinct Python ints, len >= 2
        cleaned = []
        for c in cliques:
            cl = [int(x) for x in c]
            if len(cl) >= 2 and len(set(cl)) == len(cl):
                cleaned.append(cl)
        result.append(cleaned)

    # If deadline exceeded before finishing all three, return what we have
    # (but we must return three lists; if incomplete, the result is invalid,
    #  so we rely on the fast computation finishing well before the deadline)
    return result
# EVOLVE-BLOCK-END
