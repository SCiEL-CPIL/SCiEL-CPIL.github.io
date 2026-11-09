import time
import random

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    deadline = time.time() + 78.0
    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)

    def biclique_ok(L, R):
        for u in L:
            for v in R:
                if v not in adj[u]:
                    return False
        return True

    def biclique_edges(L, R):
        return frozenset((min(u, v), max(u, v)) for u in L for v in R)

    # ---- Greedy set cover: repeatedly take a maximal biclique covering most uncovered edges ----
    all_edges = [frozenset((min(a, b), max(a, b))) for a, b in edges]
    remaining = set(all_edges)

    def grow_biclique(L, R, rem):
        # expand L/R while all cross pairs are edges; maximize |L|*|R|
        changed = True
        while changed:
            changed = False
            # try adding a vertex to L
            for u in range(n):
                if u in L:
                    continue
                if all(v in adj[u] for v in R):
                    L.add(u)
                    changed = True
            # try adding a vertex to R
            for v in range(n):
                if v in R:
                    continue
                if all(u in adj[v] for u in L):
                    R.add(v)
                    changed = True
        return L, R

    best = None
    for seed_i in range(8):
        if time.time() > deadline:
            break
        rem = set(all_edges)
        bicliques = []
        while rem:
            # choose a random edge to seed
            e = random.choice(list(rem))
            u0, v0 = tuple(e)
            L = {u0}
            R = {v0}
            L, R = grow_biclique(L, R, rem)
            cov = biclique_edges(L, R)
            # only keep if it covers at least the seed edge (it does)
            if not cov & rem:
                continue
            rem -= cov
            bicliques.append((L, R))
        if best is None or len(bicliques) < len(best):
            best = bicliques

    if best is None:
        # fallback: star cover
        return [[[u], [v for a, v in edges if a == u]] for u in range(n) if any(a == u for a, v in edges)]

    # ---- Merge pass: if two bicliques' union is a valid biclique, merge them ----
    def merge_pass(bicliques):
        improved = True
        while improved:
            improved = False
            i = 0
            while i < len(bicliques) - 1:
                if time.time() > deadline:
                    return bicliques
                j = i + 1
                while j < len(bicliques):
                    L1, R1 = bicliques[i]
                    L2, R2 = bicliques[j]
                    # candidate unions
                    cands = [
                        (L1 | L2, R1 & R2),
                        (L1 & L2, R1 | R2),
                        (L1 | L2, R1 | R2),
                    ]
                    done = False
                    for Lc, Rc in cands:
                        if not Lc or not Rc:
                            continue
                        # the merged biclique must cover exactly the union of the two edge sets
                        # and be valid (all cross pairs are edges)
                        if biclique_ok(Lc, Rc):
                            ec = biclique_edges(Lc, Rc)
                            e1 = biclique_edges(L1, R1)
                            e2 = biclique_edges(L2, R2)
                            if ec == (e1 | e2):
                                new = bicliques[:i] + bicliques[i+1:j] + [(Lc, Rc)] + bicliques[j+1:]
                                bicliques = new
                                improved = True
                                done = True
                                break
                    if done:
                        break
                    j += 1
                i += 1
        return bicliques

    best = merge_pass(best)

    # ---- Local search: try to reduce count by re-covering one biclique's edges into others ----
    # (skip if no time)
    if time.time() < deadline - 5:
        for _ in range(200):
            if time.time() > deadline:
                break
            cur = [set(L) for L, R in best]
            cur_r = [set(R) for L, R in best]
            # try dropping biclique k and covering its edges by expanding others
            k = random.randrange(len(cur))
            Lk, Rk = cur[k], cur_r[k]
            ek = biclique_edges(Lk, Rk)
            others = [i for i in range(len(cur)) if i != k]
            newL = [set(L) for L in cur]
            newR = [set(R) for R in cur_r]
            # for each edge in ek, try to absorb into a random other biclique
            ok = True
            for e in ek:
                u, v = tuple(e)
                placed = False
                for i in others:
                    # can we add u to newL[i] and v to newR[i] keeping validity?
                    if v in newR[i] and u not in newL[i] and all(x in adj[u] for x in newR[i]):
                        newL[i].add(u)
                        placed = True
                        break
                    if u in newL[i] and v not in newR[i] and all(x in adj[v] for x in newL[i]):
                        newR[i].add(v)
                        placed = True
                        break
                if not placed:
                    ok = False
                    break
            if ok:
                cand = [(newL[i], newR[i]) for i in others]
                if len(cand) < len(best):
                    best = merge_pass(cand)

    # ---- Finalize with exact types ----
    result = []
    for L, R in best:
        result.append([list(sorted(L)), list(sorted(R))])
    return result
# EVOLVE-BLOCK-END
