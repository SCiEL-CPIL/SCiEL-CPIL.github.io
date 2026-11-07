import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    t0 = time.time()
    deadline = t0 + 85.0

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

    def cross_ok(L, R, eset):
        for u in L:
            for v in R:
                if (min(u, v), max(u, v)) not in eset:
                    return False
        return True

    def find_max_biclique(remaining_edges, eset):
        """Find the largest biclique (by |L|*|R|) in the remaining edge set."""
        if not remaining_edges:
            return None, None, None
        best_L = None
        best_R = None
        best_size = 0
        # Try each edge as a seed
        seed_edges = list(remaining_edges)
        random.shuffle(seed_edges)
        for (u, v) in seed_edges:
            L = {u}
            R = {v}
            improved = True
            while improved:
                improved = False
                # Try to add to L
                for w in range(n):
                    if w in L:
                        continue
                    if all((min(w, r), max(w, r)) in eset for r in R):
                        if all((min(w, r), max(w, r)) in remaining_edges for r in R):
                            L.add(w)
                            improved = True
                # Try to add to R
                for w in range(n):
                    if w in R:
                        continue
                    if all((min(w, l), max(w, l)) in eset for l in L):
                        if all((min(w, l), max(w, l)) in remaining_edges for l in L):
                            R.add(w)
                            improved = True
            size = len(L) * len(R)
            if size > best_size:
                best_size = size
                best_L = L
                best_R = R
        if best_L is None:
            return None, None, None
        return sorted(best_L), sorted(best_R), best_size

    def greedy_max_biclique(remaining_edges, eset, deadline):
        """Repeatedly extract the largest biclique from remaining edges."""
        bics = []
        rem = set(remaining_edges)
        while rem and time.time() < deadline:
            L, R, size = find_max_biclique(rem, eset)
            if L is None or not L or not R:
                break
            # Remove covered edges
            for u in L:
                for v in R:
                    rem.discard((min(u, v), max(u, v)))
            bics.append((L, R))
        # Pack remaining edges as single-edge bicliques
        for e in rem:
            bics.append(([e[0]], [e[1]]))
        return bics

    def count(bics):
        return sum(len(L) * len(R) for (L, R) in bics)

    def valid(bics, eset):
        seen = set()
        for (L, R) in bics:
            if not L or not R:
                return False
            for u in L:
                for v in R:
                    e = (min(u, v), max(u, v))
                    if e in seen:
                        return False
                    if e not in eset:
                        return False
                    seen.add(e)
        return seen == eset

    def final(bics):
        out = []
        for (L, R) in bics:
            out.append([list(L), list(R)])
        return out

    best = None
    best_cnt = m + 1

    # ---- Multiple random restarts with max-biclique extraction ----
    random.seed(42)
    for restart in range(20):
        if time.time() >= deadline:
            break
        bics = greedy_max_biclique(Eset, Eset, deadline - 2.0)
        if not bics:
            continue
        if valid(bics, Eset):
            c = count(bics)
            if c < best_cnt:
                best_cnt = c
                best = [list(b) for b in bics]

    # ---- Local search: merge bicliques ----
    def try_merge(bics, eset, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(bics)):
                if time.time() >= deadline:
                    break
                Li, Ri = bics[i]
                for j in range(i + 1, len(bics)):
                    Lj, Rj = bics[j]
                    newL = sorted(set(Li) | set(Lj))
                    newR = sorted(set(Ri) | set(Rj))
                    if cross_ok(newL, newR, eset):
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

    # ---- Local search: edge reassignment ----
    def try_edge_reassign(bics, eset, deadline):
        """Try to reassign individual edges to reduce the number of bicliques."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # For each biclique, try removing an edge and see if it can go elsewhere
            for i in range(len(bics)):
                if time.time() >= deadline:
                    break
                Li, Ri = bics[i]
                # Try removing one vertex from L and seeing if the remaining is still valid
                for u in list(Li):
                    newL = [x for x in Li if x != u]
                    if not newL:
                        continue
                    # Check if edges from u to Ri can be absorbed by other bicliques
                    can_reassign = True
                    for v in Ri:
                        e = (min(u, v), max(u, v))
                        found = False
                        for j in range(len(bics)):
                            if j == i:
                                continue
                            Lj, Rj = bics[j]
                            if u in Lj and v in Rj:
                                found = True
                                break
                            if u in Rj and v in Lj:
                                found = True
                                break
                        if not found:
                            can_reassign = False
                            break
                    if can_reassign:
                        bics[i] = (newL, Ri)
                        if not bics[i][0]:
                            del bics[i]
                        improved = True
                        break
                if improved:
                    break
                # Try removing one vertex from R
                for v in list(Ri):
                    newR = [x for x in Ri if x != v]
                    if not newR:
                        continue
                    can_reassign = True
                    for u in Li:
                        e = (min(u, v), max(u, v))
                        found = False
                        for j in range(len(bics)):
                            if j == i:
                                continue
                            Lj, Rj = bics[j]
                            if u in Lj and v in Rj:
                                found = True
                                break
                            if u in Rj and v in Lj:
                                found = True
                                break
                        if not found:
                            can_reassign = False
                            break
                    if can_reassign:
                        bics[i] = (Li, newR)
                        if not bics[i][1]:
                            del bics[i]
                        improved = True
                        break
                if improved:
                    break
        return bics

    # ---- Local search: merge via edge overlap ----
    def try_merge_edges(bics, eset, deadline):
        """Merge two bicliques if their edge sets can form a valid biclique."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(bics)):
                if time.time() >= deadline:
                    break
                Li, Ri = bics[i]
                for j in range(i + 1, len(bics)):
                    Lj, Rj = bics[j]
                    # Try union of L sides and R sides
                    newL = sorted(set(Li) | set(Lj))
                    newR = sorted(set(Ri) | set(Rj))
                    if cross_ok(newL, newR, eset):
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

    # ---- Run local search on the best solution ----
    if best is not None:
        bics = [list(b) for b in best]
        bics = try_merge(bics, Eset, deadline - 1.0)
        bics = try_merge_edges(bics, Eset, deadline - 1.0)
        bics = try_edge_reassign(bics, Eset, deadline - 0.5)
        # Iterate
        for _ in range(5):
            if time.time() >= deadline - 0.5:
                break
            before = count(bics)
            bics = try_merge(bics, Eset, deadline - 0.5)
            bics = try_merge_edges(bics, Eset, deadline - 0.5)
            bics = try_edge_reassign(bics, Eset, deadline - 0.5)
            if count(bics) < before:
                continue
            else:
                break
        if valid(bics, Eset) and count(bics) < best_cnt:
            best_cnt = count(bics)
            best = bics

    if best is None:
        # fallback: single-edge bicliques
        best = [([u], [v]) for (u, v) in edges]

    return final(best)
