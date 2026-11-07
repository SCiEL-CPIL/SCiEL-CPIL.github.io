"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    n = P['n']
    edges = P['edges']

    # Build adjacency: adj[u] = set of neighbors
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))

    m = len(edges)

    def edge_exists(u, v):
        return (min(u, v), max(u, v)) in edge_set

    # Greedy set cover: repeatedly pick the biclique [L, R] that covers the most
    # remaining edges, where L is a small clique (size 1-3) and R = common neighbors.
    
    uncovered = set(edge_set)
    bicliques = []

    # Precompute common neighbors for all pairs
    common_neighbors = {}
    for u in range(n):
        for v in range(u + 1, n):
            if edge_exists(u, v):
                cn = adj[u] & adj[v]
                cn.discard(u)
                cn.discard(v)
                common_neighbors[(u, v)] = cn

    # Precompute common neighbors for all triples (cliques of size 3)
    common_neighbors3 = {}
    for u in range(n):
        for v in range(u + 1, n):
            if not edge_exists(u, v):
                continue
            for w in range(v + 1, n):
                if edge_exists(u, w) and edge_exists(v, w):
                    cn = adj[u] & adj[v] & adj[w]
                    cn.discard(u)
                    cn.discard(v)
                    cn.discard(w)
                    if cn:
                        common_neighbors3[(u, v, w)] = cn

    def find_best_biclique(remaining_edges):
        """Find the biclique [L, R] that covers the most edges in remaining_edges."""
        best_count = 0
        best_L = None
        best_R = None

        # Try single-vertex left sides: [u, R] where R = neighbors of u with uncovered edges
        for u in range(n):
            R = set()
            for v in adj[u]:
                e = (min(u, v), max(u, v))
                if e in remaining_edges:
                    R.add(v)
            if len(R) > best_count:
                best_count = len(R)
                best_L = [u]
                best_R = R

        # Try two-vertex left sides: [u,v, R] where R = common neighbors with both edges uncovered
        for (u, v), cn in common_neighbors.items():
            if not cn:
                continue
            R = set()
            for r in cn:
                eu = (min(u, r), max(u, r))
                ev = (min(v, r), max(v, r))
                if eu in remaining_edges and ev in remaining_edges:
                    R.add(r)
            count = len(R) * 2
            if count > best_count:
                best_count = count
                best_L = [u, v]
                best_R = R

        # Try three-vertex left sides: [u,v,w, R] where R = common neighbors with all 3 edges uncovered
        for (u, v, w), cn in common_neighbors3.items():
            if not cn:
                continue
            R = set()
            for r in cn:
                eu = (min(u, r), max(u, r))
                ev = (min(v, r), max(v, r))
                ew = (min(w, r), max(w, r))
                if eu in remaining_edges and ev in remaining_edges and ew in remaining_edges:
                    R.add(r)
            count = len(R) * 3
            if count > best_count:
                best_count = count
                best_L = [u, v, w]
                best_R = R

        return best_L, best_R, best_count

    # Iterate greedy selection
    while uncovered and time.time() < deadline:
        L, R, count = find_best_biclique(uncovered)
        if L is None or count == 0:
            # Fallback: single edge as biclique
            e = next(iter(uncovered))
            L = [e[0]]
            R = [e[1]]
            count = 1
        bicliques.append([list(L), list(R)])
        # Remove covered edges
        for l in L:
            for r in R:
                e = (min(l, r), max(l, r))
                if e in uncovered:
                    uncovered.discard(e)

    # Verify all edges are covered
    if uncovered:
        for e in list(uncovered):
            bicliques.append([[e[0]], [e[1]]])

    # Improvement phase: try to merge bicliques
    improved = True
    while improved and time.time() < deadline:
        improved = False
        new_bicliques = []
        merged = [False] * len(bicliques)
        for i in range(len(bicliques)):
            if merged[i]:
                continue
            for j in range(i + 1, len(bicliques)):
                if merged[j]:
                    continue
                Li, Ri = bicliques[i]
                Lj, Rj = bicliques[j]
                # Try merging: new_L = Li ∪ Lj, new_R = Ri ∪ Rj
                new_L_set = set(Li) | set(Lj)
                new_R_set = set(Ri) | set(Rj)
                # Check disjoint
                if new_L_set & new_R_set:
                    continue
                # Check all cross edges exist
                valid = True
                for l in new_L_set:
                    for r in new_R_set:
                        if not edge_exists(l, r):
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    # Check that merged edge set equals union of old edge sets
                    merged_edges = set()
                    for l in new_L_set:
                        for r in new_R_set:
                            merged_edges.add((min(l, r), max(l, r)))
                    old_edges = set()
                    for l in Li:
                        for r in Ri:
                            old_edges.add((min(l, r), max(l, r)))
                    for l in Lj:
                        for r in Rj:
                            old_edges.add((min(l, r), max(l, r)))
                    if merged_edges == old_edges:
                        new_bicliques.append([list(new_L_set), list(new_R_set)])
                        merged[i] = True
                        merged[j] = True
                        improved = True
                        break
            if improved and merged[i]:
                continue
            if not merged[i]:
                new_bicliques.append(bicliques[i])
        bicliques = new_bicliques

    # Final validation and type conversion
    result = []
    for L, R in bicliques:
        L_ints = [int(x) for x in L]
        R_ints = [int(x) for x in R]
        if len(L_ints) > 0 and len(R_ints) > 0:
            result.append([L_ints, R_ints])

    return result
# EVOLVE-BLOCK-END
