import time
import random
import numpy as np

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    n = P['n']
    edges = P['edges']

    def make_edge(u, v):
        return (min(u, v), max(u, v))

    # Build adjacency and edge index
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        e = make_edge(u, v)
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add(e)

    edge_list = list(edge_set)
    m = len(edge_list)
    edge_to_idx = {e: i for i, e in enumerate(edge_list)}
    idx_to_edge = edge_list

    # Incidence: bitmask of edges incident to each vertex
    inc = np.zeros(n, dtype=np.uint64)
    for (u, v), idx in edge_to_idx.items():
        inc[u] |= (np.uint64(1) << idx)
        inc[v] |= (np.uint64(1) << idx)

    # Full adjacency as bitmask (for fast biclique validity)
    adjmask = np.zeros(n, dtype=np.uint64)
    for u in range(n):
        mask = 0
        for v in adj[u]:
            mask |= (1 << v)
        adjmask[u] = np.uint64(mask)

    # Vertex mask helper
    def vmask(vs):
        mask = 0
        for v in vs:
            mask |= (1 << v)
        return np.uint64(mask)

    def bc_mask(L, R):
        """Bitmask of cross edges of biclique (L,R), assuming all are in edge_set."""
        L = list(L); R = list(R)
        mask = 0
        for l in L:
            m2 = 0
            for r in R:
                if l != r:
                    m2 |= (1 << r)
            mask |= int(adjmask[l] & m2)
        return np.uint64(mask)

    def bc_valid(L, R):
        """Check every cross pair is a host edge."""
        for l in L:
            m2 = 0
            for r in R:
                if l != r:
                    m2 |= (1 << r)
            if int(adjmask[l] & m2) != m2:
                return False
        return True

    def expand(L, R, rem):
        """Grow (L,R) to maximal biclique within residual edge set `rem` (uint64)."""
        L = set(L); R = set(R)
        changed = True
        while changed:
            changed = False
            newL = []
            for w in range(n):
                if w in L or w in R:
                    continue
                ok = True
                for r in R:
                    if make_edge(w, r) not in rem:
                        ok = False
                        break
                if ok:
                    newL.append(w)
            if newL:
                L |= set(newL)
                changed = True
            newR = []
            for w in range(n):
                if w in L or w in R:
                    continue
                ok = True
                for l in L:
                    if make_edge(w, l) not in rem:
                        ok = False
                        break
                if ok:
                    newR.append(w)
            if newR:
                R |= set(newR)
                changed = True
        return L, R

    def greedy_partition(rem, seed_order, rng):
        """Repeatedly extract largest biclique from residual `rem` (set of edges)."""
        bicliques = []
        rem = set(rem)
        while rem:
            if time.time() > deadline:
                break
            # candidate seeds
            best = None
            best_size = -1
            for (u, v) in seed_order:
                if make_edge(u, v) not in rem:
                    continue
                L, R = expand({u}, {v}, rem)
                sz = len(L) * len(R)
                if sz > best_size:
                    best_size = sz
                    best = (L, R)
            if best is None:
                # single edge fallback
                e = next(iter(rem))
                bicliques.append((frozenset([e[0]]), frozenset([e[1]])))
                rem.discard(e)
                continue
            L, R = best
            bicliques.append((frozenset(L), frozenset(R)))
            for l in L:
                for r in R:
                    if l != r:
                        rem.discard(make_edge(l, r))
        for e in rem:
            bicliques.append((frozenset([e[0]]), frozenset([e[1]])))
        return bicliques

    def try_merge(bicliques):
        """Merge any two bicliques whose union is a valid biclique."""
        cur = list(bicliques)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(cur)):
                for j in range(i + 1, len(cur)):
                    L1, R1 = cur[i]; L2, R2 = cur[j]
                    nL = L1 | L2; nR = R1 | R2
                    if nL & nR:
                        continue
                    if bc_valid(nL, nR):
                        cur[i] = (frozenset(nL), frozenset(nR))
                        cur.pop(j)
                        improved = True
                        break
                if improved:
                    break
        return cur

    def refine(bicliques):
        """Local search: merges, vertex moves, edge swaps, 3-way merges."""
        cur = list(bicliques)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # --- merges ---
            cur = try_merge(cur)
            if len(cur) < len(bicliques) or True:
                pass
            # --- vertex moves (move a vertex from one side of biclique i to same side of j) ---
            for i in range(len(cur)):
                if time.time() > deadline:
                    break
                Li, Ri = cur[i]
                for side, S in (('L', Li), ('R', Ri)):
                    for v in list(S):
                        newS = S - {v}
                        if not newS:
                            continue
                        # remaining i still valid
                        if side == 'L':
                            if not bc_valid(newS, Ri):
                                continue
                        else:
                            if not bc_valid(Li, newS):
                                continue
                        for j in range(len(cur)):
                            if i == j:
                                continue
                            Lj, Rj = cur[j]
                            if side == 'L':
                                if v in Lj or v in Rj:
                                    continue
                                if not bc_valid(Lj | {v}, Rj):
                                    continue
                                new_cur = list(cur)
                                new_cur[i] = (frozenset(newS), frozenset(Ri))
                                new_cur[j] = (frozenset(Lj | {v}), frozenset(Rj))
                            else:
                                if v in Lj or v in Rj:
                                    continue
                                if not bc_valid(Lj, Rj | {v}):
                                    continue
                                new_cur = list(cur)
                                new_cur[i] = (frozenset(Li), frozenset(newS))
                                new_cur[j] = (frozenset(Lj), frozenset(Rj | {v}))
                            merged = try_merge(new_cur)
                            if len(merged) < len(cur):
                                cur = merged
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # --- edge swaps: move one edge from biclique i to biclique j ---
            for i in range(len(cur)):
                if time.time() > deadline:
                    break
                Li, Ri = cur[i]
                for j in range(len(cur)):
                    if i == j:
                        continue
                    Lj, Rj = cur[j]
                    # candidate edges to move from i
                    for l in Li:
                        for r in Ri:
                            if l == r:
                                continue
                            e = make_edge(l, r)
                            # try adding e to j (as new cross edge)
                            # e=(a,b); need a in one side, b in other of j, or extend
                            # simplest: try inserting into j's sides
                            # check if (a in Lj and b in Rj) or (b in Lj and a in Rj) already -> then it's already covered
                            # We want to MOVE: remove from i, ensure j covers it.
                            # Determine if j can cover e by adding a vertex.
                            # Case: a in Lj, b not in Rj -> add b to Rj
                            if a in Lj and b not in Rj and b not in Lj:
                                if bc_valid(Lj, Rj | {b}):
                                    new_cur = list(cur)
                                    # remove e from i: split i? Instead remove vertex if i becomes smaller
                                    # Represent removal by deleting the edge: but biclique is full cross product.
                                    # Removing a single edge from a biclique breaks it. So skip single-edge removal.
                                    pass
            # --- 3-way merge: merge three bicliques if union valid ---
            for i in range(len(cur)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(cur)):
                    for k in range(j + 1, len(cur)):
                        L = cur[i][0] | cur[j][0] | cur[k][0]
                        R = cur[i][1] | cur[j][1] | cur[k][1]
                        if L & R:
                            continue
                        if bc_valid(L, R):
                            cur[i] = (frozenset(L), frozenset(R))
                            for idx in (k, j):
                                cur.pop(idx)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            bicliques = cur
        return cur

    def full_pipeline(seed_order, rng):
        bicliques = greedy_partition(edge_set, seed_order, rng)
        bicliques = try_merge(bicliques)
        bicliques = refine(bicliques)
        return bicliques

    # Prepare diverse seed orders
    edge_pairs = [(e[0], e[1]) for e in edge_list]
    best_result = None
    best_count = m

    # Baseline: deterministic order
    base_order = list(edge_pairs)
    res = full_pipeline(base_order, None)
    if len(res) < best_count:
        best_count = len(res)
        best_result = res

    # Random restarts
    rng = random.Random(12345)
    n_restarts = 40
    for it in range(n_restarts):
        if time.time() > deadline - 3.0:
            break
        order = list(edge_pairs)
        rng.shuffle(order)
        res = full_pipeline(order, rng)
        if len(res) < best_count:
            best_count = len(res)
            best_result = res

    if best_result is None:
        best_result = try_merge(greedy_partition(edge_set, base_order, None))

    # Final polish
    best_result = refine(best_result)

    # Convert to required format
    result = []
    for L, R in best_result:
        result.append([list(L), list(R)])

    return result
