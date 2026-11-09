"""Biclique partition solver for random graph."""
import time

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def _build_adjacency(n, edges):
    """Build adjacency sets for fast lookup."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _expand_biclique(L, R, adj, remaining_edges_set, n):
    """Expand a biclique [L, R] to be maximal with respect to remaining edges.
    
    A vertex x can be added to L if all edges (x, r) for r in R are in remaining_edges_set.
    A vertex y can be added to R if all edges (l, y) for l in L are in remaining_edges_set.
    """
    L = list(L)
    R = list(R)
    changed = True
    while changed:
        changed = False
        # Try to add vertices to L
        for x in range(n):
            if x in L:
                continue
            # Check if x connects to all vertices in R via remaining edges
            can_add = True
            for r in R:
                if (min(x, r), max(x, r)) not in remaining_edges_set:
                    can_add = False
                    break
            if can_add:
                L.append(x)
                changed = True
        # Try to add vertices to R
        for y in range(n):
            if y in R:
                continue
            # Check if y connects to all vertices in L via remaining edges
            can_add = True
            for l in L:
                if (min(l, y), max(l, y)) not in remaining_edges_set:
                    can_add = False
                    break
            if can_add:
                R.append(y)
                changed = True
    return L, R


def _find_best_biclique(remaining_edges, adj, n, deadline):
    """Find the largest biclique from the remaining edges.
    
    Try each edge as a seed, expand it, and keep the best.
    """
    if not remaining_edges:
        return None, None
    
    best_size = 0
    best_L = None
    best_R = None
    
    # Try each remaining edge as a seed
    for u, v in remaining_edges:
        if time.time() > deadline:
            break
        # Seed biclique: L={u}, R={v}
        L, R = _expand_biclique([u], [v], adj, remaining_edges, n)
        size = len(L) * len(R)
        if size > best_size:
            best_size = size
            best_L = L
            best_R = R
    
    return best_L, best_R


def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = _build_adjacency(n, edges)
    
    # Create set of all edges for fast lookup
    all_edges_set = set()
    for u, v in edges:
        all_edges_set.add((min(u, v), max(u, v)))
    
    remaining_edges = list(all_edges_set)
    bicliques = []
    
    deadline = time.time() + 80  # Safety margin
    
    while remaining_edges:
        if time.time() > deadline:
            # Fallback: create star bicliques for remaining edges
            for u, v in remaining_edges:
                bicliques.append([[u], [v]])
            break
        
        # Find the best biclique
        best_L, best_R = _find_best_biclique(remaining_edges, adj, n, deadline)
        
        if best_L is None or best_R is None:
            # Fallback
            for u, v in remaining_edges:
                bicliques.append([[u], [v]])
            break
        
        # Add this biclique
        bicliques.append([list(best_L), list(best_R)])
        
        # Remove all edges covered by this biclique from remaining
        covered = set()
        for l in best_L:
            for r in best_R:
                covered.add((min(l, r), max(l, r)))
        
        remaining_edges = [e for e in remaining_edges if e not in covered]
    
    # Ensure types are correct
    result = []
    for L, R in bicliques:
        L_ints = [int(x) for x in L]
        R_ints = [int(x) for x in R]
        result.append([L_ints, R_ints])
    
    return result
