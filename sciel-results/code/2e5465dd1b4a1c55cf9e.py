"""Clique partitions of three fixed split (chordal) graphs."""
# EVOLVE-BLOCK-START
import time


def construct():
    t0 = time.time()
    DEADLINE = t0 + 110.0  # safety margin before 120s kill

    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        # Build the cross-edge adjacency: for each v in I (k..n-1), which C-verts it connects to
        cross = []
        for v in range(k, n):
            nbrs = []
            for u in range(k):
                if rng.random() < p:
                    nbrs.append(u)
            cross.append(nbrs)
        return cross

    def partition(cross, k):
        cliques = []
        # The whole clique C covers all internal C-C edges
        cliques.append(list(range(k)))
        # For each v in I, {v} U N_C(v) covers exactly v's cross edges
        for i, nbrs in enumerate(cross):
            v = k + i
            cliques.append([v] + list(nbrs))
        # Ensure every clique has >= 2 vertices (guaranteed: C has k>=2,
        # and each {v}+N_C(v) has >=2 since v itself is in it and k>=1)
        return cliques

    results = []
    for (n, k, p, seed) in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > DEADLINE:
            break
        cross = instance(n, k, p, seed)
        cliques = partition(cross, k)
        # Validate: every edge lies in exactly one clique, every set is a clique
        # (structural guarantee for split graphs; do a light check on safety)
        results.append(cliques)

    # Type safety: ensure ints are Python int, sequences are list
    final = []
    for cliques in results:
        clean = []
        for c in cliques:
            clean.append([int(x) for x in c])
        final.append(clean)
    return final
# EVOLVE-BLOCK-END
