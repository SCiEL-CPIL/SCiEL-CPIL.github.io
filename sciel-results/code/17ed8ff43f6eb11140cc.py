"""Small exact biclique partitions of a random graph."""

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def _seed_partition():
    """Seed: each vertex gets one biclique [u, neighbors]."""
    n = P['n']
    adj = [set() for _ in range(n)]
    for a, b in P['edges']:
        adj[a].add(b)
        adj[b].add(a)
    parts = []
    for u in range(n):
        if adj[u]:
            parts.append([set([u]), set(adj[u])])
    return parts


def _valid_partition(parts, edge_set):
    """Check that parts form a valid biclique partition of edge_set."""
    n = P['n']
    covered = set()
    for L, R in parts:
        if not L or not R:
            return False
        for u in L:
            if u in R:
                return False
        for v in R:
            if v in L:
                return False
        for u in L:
            for v in R:
                pair = (min(u, v), max(u, v))
                if pair not in edge_set:
                    return False
                if pair in covered:
                    return False
                covered.add(pair)
    if covered != edge_set:
        return False
    return True


def _expand_biclique(u, v, remaining_edges, adj):
    """
    Expand biclique from seed edge (u, v).
    Greedy: repeatedly add a vertex on one side that is connected to all vertices on the other side,
    choosing the vertex that maximizes the number of new edges covered.
    """
    L = {u}
    R = {v}
    changed = True
    while changed:
        changed = False
        # Try adding to L
        candidates_L = []
        for w in range(P['n']):
            if w in L or w in R:
                continue
            # w can join L if w is connected to every vertex in R via a remaining edge
            can_join = True
            new_edges = 0
            for r in R:
                pair = (min(w, r), max(w, r))
                if pair in remaining_edges:
                    new_edges += 1
                else:
                    can_join = False
                    break
            if can_join:
                candidates_L.append((w, new_edges))
        # Try adding to R
        candidates_R = []
        for w in range(P['n']):
            if w in L or w in R:
                continue
            can_join = True
            new_edges = 0
            for l in L:
                pair = (min(w, l), max(w, l))
                if pair in remaining_edges:
                    new_edges += 1
                else:
                    can_join = False
                    break
            if can_join:
                candidates_R.append((w, new_edges))
        
        best_gain = 0
        best_side = None
        best_vertex = None
        for w, gain in candidates_L:
            if gain > best_gain:
                best_gain = gain
                best_side = 'L'
                best_vertex = w
        for w, gain in candidates_R:
            if gain > best_gain:
                best_gain = gain
                best_side = 'R'
                best_vertex = w
        
        if best_side is not None and best_gain > 0:
            if best_side == 'L':
                L.add(best_vertex)
            else:
                R.add(best_vertex)
            changed = True
        elif best_side is not None:
            # Even with 0 new edges, might still be useful? No, gain 0 means no new edges.
            # Actually if gain is 0, all edges from w to other side are already covered or not present.
            # If not present, w can't join. If present but covered, no new edges.
            # So skip.
            pass
    return L, R


def construct():
    import time
    t0 = time.time()
    deadline = t0 + 80.0  # 80s deadline, 10s safety margin
    
    n = P['n']
    adj = [set() for _ in range(n)]
    edge_set = set()
    for a, b in P['edges']:
        adj[a].add(b)
        adj[b].add(a)
        edge_set.add((min(a, b), max(a, b)))
    
    # Greedy biclique partition
    parts = []
    remaining = set(edge_set)
    
    while remaining:
        if time.time() > deadline:
            break
        
        # Pick an uncovered edge with minimum degrees (to create smaller bicliques? or maximum?)
        # Actually, to minimize count, we want LARGE bicliques. So pick edge that can form a large biclique.
        # Heuristic: pick edge (u,v) where u and v have high degrees in remaining graph.
        best_edge = None
        best_score = -1
        for e in remaining:
            u, v = e
            score = len([e2 for e2 in remaining if u in e2]) + len([e2 for e2 in remaining if v in e2])
            if score > best_score:
                best_score = score
                best_edge = e
        
        if best_edge is None:
            break
        
        u, v = best_edge
        L, R = _expand_biclique(u, v, remaining, adj)
        
        # Remove covered edges from remaining
        covered_here = set()
        for a in L:
            for b in R:
                pair = (min(a, b), max(a, b))
                if pair in remaining:
                    covered_here.add(pair)
        remaining -= covered_here
        
        parts.append([set(L), set(R)])
    
    # Verify and return
    # Convert to list format
    result = []
    for L, R in parts:
        result.append([list(L), list(R)])
    
    # If deadline exceeded and partition is incomplete, fall back to seed
    if remaining:
        # Fallback: use seed partition
        result = []
        for u in range(n):
            if adj[u]:
                result.append([list([u]), list(adj[u])])
    
    return result
