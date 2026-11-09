import time
import random
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill

    n = P['n']
    edges = P['edges']

    def make_edge(u, v):
        return (min(u, v), max(u, v))

    # Build adjacency sets and edge set
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        e = make_edge(u, v)
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add(e)

    total_edges = len(edge_set)

    def biclique_edges(L, R):
        out = set()
        for l in L:
            for r in R:
                if l != r:
                    out.add(make_edge(l, r))
        return out

    def find_max_biclique(remaining, adj, n, strategy='expand'):
        if not remaining:
            return None

        best_biclique = None
        best_size = 0

        def verify(L, R):
            for l in L:
                for r in R:
                    if l == r:
                        continue
                    if make_edge(l, r) not in remaining:
                        return False
            return True

        if strategy == 'expand':
            for (u, v) in remaining:
                L = {u}
                R = {v}
                changed = True
                while changed:
                    changed = False
                    new_L = set()
                    for w in range(n):
                        if w in L or w in R:
                            continue
                        valid = True
                        for r in R:
                            if make_edge(w, r) not in remaining:
                                valid = False
                                break
                        if valid:
                            new_L.add(w)
                    if new_L:
                        L |= new_L
                        changed = True
                    new_R = set()
                    for w in range(n):
                        if w in L or w in R:
                            continue
                        valid = True
                        for l in L:
                            if make_edge(w, l) not in remaining:
                                valid = False
                                break
                        if valid:
                            new_R.add(w)
                    if new_R:
                        R |= new_R
                        changed = True
                if verify(L, R) and len(L) * len(R) > best_size:
                    best_size = len(L) * len(R)
                    best_biclique = (L, R)
        elif strategy == 'common_neighbors':
            for u in range(n):
                neighbors = set()
                for v in range(n):
                    if v != u and make_edge(u, v) in remaining:
                        neighbors.add(v)
                if not neighbors:
                    continue
                R = set(neighbors)
                L = {u}
                changed = True
                while changed:
                    changed = False
                    new_L = set()
                    for w in range(n):
                        if w in L or w in R:
                            continue
                        valid = True
                        for r in R:
                            if make_edge(w, r) not in remaining:
                                valid = False
                                break
                        if valid:
                            new_L.add(w)
                    if new_L:
                        L |= new_L
                        changed = True
                    new_R = set()
                    for w in range(n):
                        if w in L or w in R:
                            continue
                        valid = True
                        for l in L:
                            if make_edge(w, l) not in remaining:
                                valid = False
                                break
                        if valid:
                            new_R.add(w)
                    if new_R:
                        R |= new_R
                        changed = True
                if verify(L, R) and len(L) * len(R) > best_size:
                    best_size = len(L) * len(R)
                    best_biclique = (L, R)
        elif strategy == 'degree_heuristic':
            deg = {}
            for v in range(n):
                deg[v] = 0
            for (u, v) in remaining:
                deg[u] += 1
                deg[v] += 1
            for v in sorted(range(n), key=lambda x: -deg[x]):
                if deg[v] == 0:
                    continue
                neighbors = set()
                for w in range(n):
                    if w != v and make_edge(v, w) in remaining:
                        neighbors.add(w)
                if not neighbors:
                    continue
                R = set(neighbors)
                L = {v}
                changed = True
                while changed:
                    changed = False
                    new_L = set()
                    for w in range(n):
                        if w in L or w in R:
                            continue
                        valid = True
                        for r in R:
                            if make_edge(w, r) not in remaining:
                                valid = False
                                break
                        if valid:
                            new_L.add(w)
                    if new_L:
                        L |= new_L
                        changed = True
                    new_R = set()
                    for w in range(n):
                        if w in L or w in R:
                            continue
                        valid = True
                        for l in L:
                            if make_edge(w, l) not in remaining:
                                valid = False
                                break
                        if valid:
                            new_R.add(w)
                    if new_R:
                        R |= new_R
                        changed = True
                if verify(L, R) and len(L) * len(R) > best_size:
                    best_size = len(L) * len(R)
                    best_biclique = (L, R)
        return best_biclique

    def greedy_partition(remaining_edges, adj, n, strategy='expand'):
        bicliques = []
        remaining = set(remaining_edges)
        while remaining:
            if time.time() > deadline:
                break
            bc = find_max_biclique(remaining, adj, n, strategy)
            if bc is None:
                for (u, v) in remaining:
                    bicliques.append((frozenset([u]), frozenset([v])))
                remaining = set()
                break
            L, R = bc
            bicliques.append((frozenset(L), frozenset(R)))
            to_remove = biclique_edges(L, R)
            remaining -= to_remove
        return bicliques, remaining

    def try_merge_all(bicliques):
        improved = True
        current = list(bicliques)
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(current)):
                    L1, R1 = current[i]
                    L2, R2 = current[j]
                    newL = L1 | L2
                    newR = R1 | R2
                    if newL & newR:
                        continue
                    valid = True
                    for l in newL:
                        for r in newR:
                            if l == r:
                                continue
                            if make_edge(l, r) not in edge_set:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        current[i] = (newL, newR)
                        current.pop(j)
                        improved = True
                        break
                if improved:
                    break
        return current

    # Try multiple strategies, keep the best partition
    best_result = None
    best_count = total_edges

    for strat in ('expand', 'common_neighbors', 'degree_heuristic'):
        if time.time() > deadline:
            break
        bicliques, leftover = greedy_partition(edge_set, adj, n, strat)
        if leftover:
            for (u, v) in leftover:
                bicliques.append((frozenset([u]), frozenset([v])))
        merged = try_merge_all(bicliques)
        if len(merged) < best_count:
            best_count = len(merged)
            best_result = merged

    if best_result is None:
        bicliques, leftover = greedy_partition(edge_set, adj, n, 'expand')
        for (u, v) in leftover:
            bicliques.append((frozenset([u]), frozenset([v])))
        best_result = try_merge_all(bicliques)

    # Local search: try to move edges from one biclique to another to merge/absorb
    current = list(best_result)

    def partition_edges(bc_list):
        # map each graph edge -> index of biclique covering it
        cov = {}
        for idx, (L, R) in enumerate(bc_list):
            for l in L:
                for r in R:
                    if l != r:
                        cov[make_edge(l, r)] = idx
        return cov

    def local_search(bicliques):
        improved = True
        cur = list(bicliques)
        while improved and time.time() < deadline:
            improved = False
            cov = partition_edges(cur)
            # For each pair of bicliques, try to move an edge from A to B if B can absorb it
            for i in range(len(cur)):
                if time.time() > deadline:
                    break
                for j in range(len(cur)):
                    if i == j:
                        continue
                    Li, Ri = cur[i]
                    Lj, Rj = cur[j]
                    # edges currently in i
                    ei = biclique_edges(Li, Ri)
                    # try moving each edge of i into j
                    moved = False
                    for e in list(ei):
                        u, v = e
                        # can j absorb edge e? i.e. is (u,v) a cross pair within (Lj,Rj) or extendable
                        # simplest: check if u and v are both in Lj|Lj and form cross with existing
                        # We'll just try: new biclique j = (Lj ∪ {u}, Rj ∪ {v}) if u not in Rj, v not in Lj
                        if u in Rj or v in Lj:
                            continue
                        # try orientation (u in L, v in R)
                        candL = Lj | {u}
                        candR = Rj | {v}
                        if candL & candR:
                            continue
                        ok = True
                        for l in candL:
                            for r in candR:
                                if l == r:
                                    continue
                                if make_edge(l, r) not in edge_set:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            continue
                        # removing edge e from i: i still valid (subset of biclique is biclique)
                        # new i: remove e. We can just recompute i as biclique minus that edge's contribution
                        # Simplest: keep i as is but we must ensure e is no longer covered by i.
                        # Represent i by its vertex sets; removing a single edge is hard. Instead,
                        # rebuild i as the maximal biclique in remaining edges after removing e.
                        # To keep it simple: only accept move if i becomes empty (i.e., i == {e}).
                        if len(Li) == 1 and len(Ri) == 1:
                            # i is a single edge; move it fully into j
                            cur[j] = (candL, candR)
                            cur.pop(i)
                            improved = True
                            moved = True
                            break
                    if moved:
                        break
                if improved:
                    break
            if improved:
                cur = try_merge_all(cur)
        return cur

    if time.time() < deadline:
        current = local_search(current)
        if len(current) < best_count:
            best_count = len(current)
            best_result = current

    # Final merge pass
    best_result = try_merge_all(best_result)

    # Convert to required format
    result = []
    for L, R in best_result:
        result.append([list(L), list(R)])

    return result
