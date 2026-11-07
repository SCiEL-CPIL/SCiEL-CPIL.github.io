"""Clique partitions of three fixed split graphs (structural rewrite)."""
import time
from random import Random

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def safe_partition(edges):
        return [[int(u), int(v)] for (u, v) in edges]

    def split_partition(n, k, p, seed):
        rng = Random(seed)
        cross = {}
        for v in range(k, n):
            nb = []
            for u in range(k):
                if rng.random() < p:
                    nb.append(u)
            cross[v] = nb

        C = list(range(k))
        I = list(range(k, n))
        # order I by increasing neighbourhood size (hardest to place first)
        I.sort(key=lambda v: len(cross[v]))

        cliques = []
        for v in I:
            nb = cross[v]
            if not nb:
                continue
            placed = False
            for cli in cliques:
                Cset = [x for x in cli if x < k]
                if Cset and all(c in nb for c in Cset):
                    cli.append(v)
                    placed = True
                    break
            if not placed:
                cliques.append([v] + nb)

        if k >= 2:
            cliques.append(C)
        return cliques

    def validate(cliques, edges):
        eset = set(edges)
        seen = set()
        for cli in cliques:
            if not isinstance(cli, (list, tuple)):
                return False
            if len(cli) < 2:
                return False
            vals = list(cli)
            if any(not isinstance(x, int) or isinstance(x, bool) for x in vals):
                return False
            if len(set(vals)) != len(vals):
                return False
            for i in range(len(vals)):
                for j in range(i + 1, len(vals)):
                    a, b = vals[i], vals[j]
                    e = (a, b) if a < b else (b, a)
                    if e not in eset:
                        return False
                    if e in seen:
                        return False
                    seen.add(e)
        return seen == eset

    results = []
    for (n, k, p, seed) in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline - 5.0:
            results.append(safe_partition(instance(n, k, p, seed)))
            continue
        edges = instance(n, k, p, seed)
        try:
            cl = split_partition(n, k, p, seed)
            if not validate(cl, edges):
                cl = safe_partition(edges)
        except Exception:
            cl = safe_partition(edges)
        results.append(cl)
    return results
# EVOLVE-BLOCK-END
