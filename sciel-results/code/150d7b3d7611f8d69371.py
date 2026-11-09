import time
import random

def construct():
    t0 = time.time()
    DEADLINE = t0 + 135.0  # safety margin under the 150s kill

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def solve(edges):
        n = len(edges)
        if n == 0:
            return []
        maxv = max(v for e in edges for v in e)
        adj = [set() for _ in range(maxv + 1)]
        for u, v in edges:
            adj[u].add(v); adj[v].add(u)
        # induced-neighbourhood of an edge (endpoints + their neighbours)
        nb = [adj[u] | adj[v] | {u, v} for (u, v) in edges]

        def valid(col):
            seen = {}
            for i, c in enumerate(col):
                if c not in seen:
                    seen[c] = i
                else:
                    if nb[seen[c]] & nb[i]:
                        return False
            return True

        # ---- greedy initial colouring (colour order = edge order) ----
        col = [0] * n
        for i in range(n):
            used = set()
            nbi = nb[i]
            for j in range(i):
                if nbi & nb[j]:
                    used.add(col[j])
            c = 0
            while c in used:
                c += 1
            col[i] = c

        maxc = max(col)

        # Try to eliminate the current top colour class `maxc` by moving its
        # edges down into 0..maxc-1 (keeping the colouring valid throughout).
        def eliminate(col, top, rng, budget):
            # edges currently coloured `top`
            freed = [i for i in range(n) if col[i] == top]
            if not freed:
                return True
            # temporarily uncolour them
            for i in freed:
                col[i] = -1
            # now recolour each freed edge into a colour < top, trying to keep
            # validity; use a simple backtracking-free greedy with retries.
            # We only need a colour in 0..top-1 that does not clash with any
            # edge whose induced-neighbourhood intersects this edge's.
            def assign_freed(order, rng, budget):
                for i in order:
                    nbi = nb[i]
                    forb = set()
                    for j in range(n):
                        if j != i and col[j] != -1 and (nbi & nb[j]):
                            forb.add(col[j])
                    cand = [c for c in range(top) if c not in forb]
                    if not cand:
                        return False
                    col[i] = cand[rng.randrange(len(cand))]
                return True

            # several random orders
            for _ in range(40):
                if budget() <= 0:
                    return False
                order = freed[:]
                rng.shuffle(order)
                if assign_freed(order, rng, budget):
                    if valid(col):
                        return True
                # reset freed to -1 before retry
                for i in freed:
                    col[i] = -1
            for i in freed:
                col[i] = -1
            return False

        def budget():
            return time.time() < DEADLINE

        best = col[:]
        rng = random.Random(12345)
        while budget() > 0 and maxc > 0:
            trial = best[:]
            if eliminate(trial, maxc, rng, budget):
                best = trial
                maxc -= 1
            else:
                # give up on this class but keep improving lower classes
                # by attempting to eliminate again with fresh randomness
                trial = best[:]
                rng = random.Random(random.randrange(1 << 30))
                if eliminate(trial, maxc, rng, budget):
                    best = trial
                    maxc -= 1
                else:
                    break
        return best

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        edges = instance(*spec)
        out.append(solve(edges))
    return out
