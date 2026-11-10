import time

def construct():
    t0 = time.time()
    deadline = t0 + 110.0
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition(n, k, p, seed):
        rng = Random(seed)
        cross = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross.append((v, u))
        cliques = [list(range(k))]  # core
        for (v, u) in cross:
            cliques.append([v, u])
        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    result = []
    for spec in specs:
        if time.time() > deadline:
            result.append([list(e) for e in instance(*spec)])
        else:
            result.append(partition(*spec))
    final = [[[int(x) for x in c] for c in cliques] for cliques in result]
    return final
