def construct():
    import time
    import random
    from collections import Counter

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve(n, k, p, seed, deadline):
        # Build the split graph deterministically (identical to instance()).
        rng = random.Random(seed)
        cross = {}  # i in I -> list of c in C with edge (c,i)
        for v in range(k, n):
            lst = []
            for u in range(k):
                if rng.random() < p:
                    lst.append(u)
            cross[v] = lst
        # Edges that must be covered: all edges inside C plus every cross edge.
        need = Counter()
        for u in range(k):
            for v in range(u + 1, k):
                need[(u, v)] += 1
        for i in range(k, n):
            for c in cross[i]:
                need[(c, i)] += 1
        if not need:
            return []
        # Candidate cliques: the whole clique C, and {i}+N_C(i) for each i with edges.
        cand = []
        if k >= 2:
            cand.append(list(range(k)))
        for i in range(k, n):
            if cross[i]:
                cand.append([i] + list(cross[i]))
        # Fast greedy set-cover using integer counters (each pick is O(n)).
        rem = [0] * len(cand)
        for idx, cl in enumerate(cand):
            s = 0
            for a in cl:
                for b in cl:
                    if a < b:
                        s += need[(a, b)]
            rem[idx] = s
        used = [False] * len(cand)
        chosen = []
        while True:
            best = -1
            bestval = 0
            for idx in range(len(cand)):
                if not used[idx] and rem[idx] > bestval:
                    bestval = rem[idx]
                    best = idx
            if best < 0:
                break
            used[best] = True
            chosen.append(best)
            cl = cand[best]
            for a in cl:
                for b in cl:
                    if a < b:
                        e = (a, b)
                        if need[e] > 0:
                            need[e] -= 1
                            for j in range(len(cand)):
                                if j != best and not used[j] and e in cand[j]:
                                    rem[j] -= 1
        # Cheap local improvement: drop any chosen clique that covers no edge.
        chosen = [i for i in chosen if any(need[(a, b)] > 0 for a in cand[i] for b in cand[i] if a < b)]
        out = [cand[i] for i in chosen]
        # Guarantee validity (must cover every edge); if somehow not, fall back to edges.
        # (Greedy is correct, so this is just a safety net.)
        left = set()
        for i in range(k, n):
            for c in cross[i]:
                left.add((c, i))
        for u in range(k):
            for v in range(u + 1, k):
                left.add((u, v))
        covered = set()
        for cl in out:
            for a in cl:
                for b in cl:
                    if a < b:
                        covered.add((a, b))
        missing = left - covered
        if missing:
            out = [[a, b] for (a, b) in sorted(missing)]
        return out

    deadline = time.time() + 130.0
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    result = []
    for spec in specs:
        n, k, p, seed = spec
        result.append(solve(n, k, p, seed, deadline))
        if time.time() > deadline - 2.0:
            break
    # Pad any shortfalls with a trivial valid (but large) edge partition.
    while len(result) < 3:
        n, k, p, seed = specs[len(result)]
        result.append([list(e) for e in instance(n, k, p, seed)])
    return result
