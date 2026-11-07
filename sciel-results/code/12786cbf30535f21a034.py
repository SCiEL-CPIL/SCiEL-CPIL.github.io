import time

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    t0 = time.time()
    deadline = t0 + 80.0

    n = int(P['n'])
    raw = P['edges']
    edges = []
    for e in raw:
        u = int(e[0]); v = int(e[1])
        if u > v:
            u, v = v, u
        edges.append((u, v))
    Eset = set(edges)
    m = len(edges)
    if m == 0:
        return []

    # adjacency sets
    adj = [set() for _ in range(n)]
    for (u, v) in edges:
        adj[u].add(v); adj[v].add(u)

    def cross_ok(L, R):
        for u in L:
            for v in R:
                if (min(u, v), max(u, v)) not in Eset:
                    return False
        return True

    def make_biclique(seed_u, seed_v, remaining):
        L = {seed_u}
        R = {seed_v}
        improved = True
        while improved:
            improved = False
            candL = set()
            for w in range(n):
                if w in L:
                    continue
                if all((min(w, r), max(w, r)) in remaining for r in R):
                    candL.add(w)
            for w in candL:
                L.add(w); improved = True
            candR = set()
            for w in range(n):
                if w in R:
                    continue
                if all((min(w, l), max(w, l)) in remaining for l in L):
                    candR.add(w)
            for w in candR:
                R.add(w); improved = True
        L = sorted(L); R = sorted(R)
        # remove edges
        for u in L:
            for v in R:
                remaining.discard((min(u, v), max(u, v)))
        return (L, R)

    def greedy(remaining, order):
        bics = []
        rem = set(remaining)
        for (u, v) in order:
            if (u, v) not in rem:
                continue
            L, R = make_biclique(u, v, rem)
            bics.append((L, R))
        return bics, rem

    def count(bics):
        return sum(len(L) * len(R) for (L, R) in bics)

    def valid(bics):
        seen = set()
        for (L, R) in bics:
            if not L or not R:
                return False
            for u in L:
                for v in R:
                    e = (min(u, v), max(u, v))
                    if e in seen:
                        return False
                    if e not in Eset:
                        return False
                    seen.add(e)
        return seen == Eset

    def final(bics):
        out = []
        for (L, R) in bics:
            out.append([list(L), list(R)])
        return out

    best = None
    best_cnt = m + 1

    # ---- initial solutions with different orderings ----
    candidates = []
    # order by increasing degree
    deg = [len(adj[i]) for i in range(n)]
    ord1 = sorted(edges, key=lambda e: (deg[e[0]] + deg[e[1]], e))
    ord2 = sorted(edges, key=lambda e: (-deg[e[0]] - deg[e[1]], e))
    ord3 = sorted(edges, key=lambda e: (min(deg[e[0]], deg[e[1]]), e))
    for order in (ord1, ord2, ord3, edges):
        bics, rem = greedy(Eset, order)
        if rem:
            # pack leftovers as single edges
            for e in rem:
                bics.append(([e[0]], [e[1]]))
        c = count(bics)
        if valid(bics) and c < best_cnt:
            best_cnt = c
            best = [b for b in bics]

    # ---- local search: merge two bicliques if it reduces the count ----
    def try_merge(bics):
        improved = True
        while improved and (time.time() < deadline):
            improved = False
            for i in range(len(bics)):
                if time.time() >= deadline:
                    break
                Li, Ri = bics[i]
                for j in range(i + 1, len(bics)):
                    Lj, Rj = bics[j]
                    if len(Li) * len(Ri) + len(Lj) * len(Rj) <= 1:
                        continue
                    newL = sorted(set(Li) | set(Lj))
                    newR = sorted(set(Ri) | set(Rj))
                    if cross_ok(newL, newR):
                        newc = len(newL) * len(newR)
                        oldc = len(Li) * len(Ri) + len(Lj) * len(Rj)
                        if newc < oldc:
                            bics[i] = (newL, newR)
                            del bics[j]
                            improved = True
                            break
                if improved:
                    break
        return bics

    # ---- local search: move a vertex from one side into the other biclique ----
    def try_move(bics):
        improved = True
        while improved and (time.time() < deadline):
            improved = False
            for i in range(len(bics)):
                if time.time() >= deadline:
                    break
                Li, Ri = bics[i]
                # try moving a vertex of Li to Ri (i.e., to the other side of same biclique)
                for u in list(Li):
                    newL = [x for x in Li if x != u]
                    newR = sorted(set(Ri) | {u})
                    if not newL:
                        continue
                    if cross_ok(newL, newR):
                        oldc = len(Li) * len(Ri)
                        newc = len(newL) * len(newR)
                        if newc < oldc:
                            bics[i] = (newL, newR)
                            improved = True
                            break
                if improved:
                    break
                # try moving a vertex of Ri to Li
                for v in list(Ri):
                    newR = [x for x in Ri if x != v]
                    newL = sorted(set(Li) | {v})
                    if not newR:
                        continue
                    if cross_ok(newL, newR):
                        oldc = len(Li) * len(Ri)
                        newc = len(newL) * len(newR)
                        if newc < oldc:
                            bics[i] = (newL, newR)
                            improved = True
                            break
                if improved:
                    break
        return bics

    # ---- local search: move a vertex between two bicliques ----
    def try_relocate(bics):
        improved = True
        while improved and (time.time() < deadline):
            improved = False
            for i in range(len(bics)):
                if time.time() >= deadline:
                    break
                Li, Ri = bics[i]
                for u in list(Li):
                    for j in range(len(bics)):
                        if j == i:
                            continue
                        Lj, Rj = bics[j]
                        # move u from Li to Lj
                        newLi = [x for x in Li if x != u]
                        newLj = sorted(set(Lj) | {u})
                        if not newLi:
                            continue
                        if cross_ok(newLi, Ri) and cross_ok(newLj, Rj):
                            oldc = len(Li) * len(Ri) + len(Lj) * len(Rj)
                            newc = len(newLi) * len(Ri) + len(newLj) * len(Rj)
                            if newc < oldc:
                                bics[i] = (newLi, Ri)
                                bics[j] = (newLj, Rj)
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
                # move from Ri to Rj
                for v in list(Ri):
                    for j in range(len(bics)):
                        if j == i:
                            continue
                        Lj, Rj = bics[j]
                        newRi = [x for x in Ri if x != v]
                        newRj = sorted(set(Rj) | {v})
                        if not newRi:
                            continue
                        if cross_ok(Li, newRi) and cross_ok(Lj, newRj):
                            oldc = len(Li) * len(Ri) + len(Lj) * len(Rj)
                            newc = len(Li) * len(newRi) + len(Lj) * len(newRj)
                            if newc < oldc:
                                bics[i] = (Li, newRi)
                                bics[j] = (Lj, newRj)
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        return bics

    # ---- local search: split a biclique into two smaller ones that cover same edges ----
    def try_split(bics):
        improved = True
        while improved and (time.time() < deadline):
            improved = False
            for i in range(len(bics)):
                if time.time() >= deadline:
                    break
                L, R = bics[i]
                if len(L) < 2 or len(R) < 2:
                    continue
                # try to find a partition L=A∪B, R=C∪D such that A×C and B×D are bicliques
                # and A×D, B×C have no edges (so the union of two bicliques exactly covers L×R)
                best_split = None
                best_split_c = len(L) * len(R)
                for a in L:
                    for b in L:
                        if a >= b:
                            continue
                        A = [a]
                        B = [x for x in L if x != a]
                        # determine C: vertices in R adjacent to all of A and not to any of B
                        C = []
                        D = []
                        ok = True
                        for v in R:
                            inC = all((min(a, v), max(a, v)) in Eset for a in A)
                            inD = all((min(b2, v), max(b2, v)) in Eset for b2 in B)
                            if inC and not inD:
                                C.append(v)
                            elif inD and not inC:
                                D.append(v)
                            elif inC and inD:
                                # v adjacent to both A and B; must go somewhere; try C
                                C.append(v)
                            else:
                                ok = False
                                break
                        if not ok:
                            continue
                        if not C or not D:
                            continue
                        # verify A×C and B×D are bicliques
                        if cross_ok(A, C) and cross_ok(B, D):
                            # verify A×D and B×C have no edges (to keep exact partition)
                            ok2 = True
                            for a2 in A:
                                for d2 in D:
                                    if (min(a2, d2), max(a2, d2)) in Eset:
                                        ok2 = False
                                        break
                                if not ok2:
                                    break
                            if ok2:
                                for b2 in B:
                                    for c2 in C:
                                        if (min(b2, c2), max(b2, c2)) in Eset:
                                            ok2 = False
                                            break
                                        if not ok2:
                                            break
                            if ok2:
                                c = len(A) * len(C) + len(B) * len(D)
                                if c < best_split_c:
                                    best_split_c = c
                                    best_split = (A, C, B, D)
                if best_split is not None:
                    A, C, B, D = best_split
                    bics[i] = (sorted(A), sorted(C))
                    bics.insert(i + 1, (sorted(B), sorted(D)))
                    improved = True
                    break
        return bics

    # ---- run local search on the best solution ----
    if best is not None:
        bics = [list(b) for b in best]
        bics = try_merge(bics)
        bics = try_move(bics)
        bics = try_relocate(bics)
        bics = try_split(bics)
        # iterate a few times
        for _ in range(3):
            if time.time() >= deadline:
                break
            before = count(bics)
            bics = try_merge(bics)
            bics = try_move(bics)
            bics = try_relocate(bics)
            bics = try_split(bics)
            if count(bics) < before:
                continue
            else:
                break
        if valid(bics) and count(bics) < best_cnt:
            best_cnt = count(bics)
            best = bics

    if best is None:
        # fallback: single-edge bicliques
        best = [([u], [v]) for (u, v) in edges]

    return final(best)
