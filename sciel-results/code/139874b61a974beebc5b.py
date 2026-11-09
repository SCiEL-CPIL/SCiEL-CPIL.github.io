"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency structure
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u,v), max(u,v)))
    
    # We need to partition all edges into bicliques.
    # A biclique (L, R) covers edges {(u,v) : u in L, v in R} where all such pairs are in edge_set.
    # The covered edges must be disjoint across bicliques and cover all edges.
    
    # Greedy approach: repeatedly find a biclique that covers many uncovered edges.
    # Start with all edges uncovered.
    
    remaining_edges = set(edge_set)
    bicliques = []
    
    # To find good bicliques, we can try: for each pair of vertices (u,v) that are adjacent,
    # try to build a biclique containing edge (u,v) by expanding L and R.
    
    # A simpler heuristic: try bicliques of the form ({u}, N(u) ∩ some set) or ({u,v}, ...)
    # Let's try a greedy set cover:
    # At each step, find the biclique (L,R) that maximizes |{(u,v) in L×R : (u,v) in remaining_edges}|
    # subject to L×R ⊆ edge_set (all cross pairs are edges).
    
    # To make this tractable, we'll consider candidate bicliques generated from pairs of vertices.
    
    def get_biclique_edges(L, R):
        """Return set of edges (as frozenset or sorted tuple) covered by biclique (L,R)."""
        es = set()
        for u in L:
            for v in R:
                e = (min(u,v), max(u,v))
                es.add(e)
        return es
    
    def is_valid_biclique(L, R):
        """Check if all cross pairs in L×R are edges in the original graph."""
        for u in L:
            for v in R:
                e = (min(u,v), max(u,v))
                if e not in edge_set:
                    return False
        return True
    
    # Generate candidate bicliques from pairs of uncovered edges
    # For efficiency, we'll try bicliques where L and R are small sets.
    
    # Strategy: for each uncovered edge (u,v), try to form a biclique with u in L and v in R,
    # then try to expand L by adding vertices that are adjacent to all of R,
    # and expand R by adding vertices adjacent to all of L.
    
    def expand_biclique(L, R, remaining):
        """Try to expand L and R while maintaining validity and covering remaining edges."""
        L = set(L)
        R = set(R)
        changed = True
        while changed:
            changed = False
            # Try to add vertices to L
            for w in range(n):
                if w in L or w in R:
                    continue
                # Check if w can be added to L: w must be adjacent to all in R
                can_add = True
                for v in R:
                    e = (min(w,v), max(w,v))
                    if e not in edge_set:
                        can_add = False
                        break
                if can_add:
                    # Check if adding w to L covers any remaining edges
                    new_edges = set()
                    for v in R:
                        e = (min(w,v), max(w,v))
                        if e in remaining:
                            new_edges.add(e)
                    if new_edges:
                        L.add(w)
                        changed = True
            # Try to add vertices to R
            for w in range(n):
                if w in L or w in R:
                    continue
                can_add = True
                for u in L:
                    e = (min(w,u), max(w,u))
                    if e not in edge_set:
                        can_add = False
                        break
                if can_add:
                    new_edges = set()
                    for u in L:
                        e = (min(w,u), max(w,u))
                        if e in remaining:
                            new_edges.add(e)
                    if new_edges:
                        R.add(w)
                        changed = True
        return L, R
    
    # Greedy: repeatedly pick the biclique covering most remaining edges
    import itertools
    
    max_iter = 1000
    for _ in range(max_iter):
        if not remaining_edges:
            break
        
        best_biclique = None
        best_count = 0
        
        # Try candidates based on each remaining edge
        for (u, v) in list(remaining_edges):
            # Try biclique with L={u}, R={v} and expand
            L, R = expand_biclique({u}, {v}, remaining_edges)
            if not L or not R:
                continue
            if L & R:
                continue
            covered = get_biclique_edges(L, R)
            # Only count edges that are in remaining
            covered_remaining = covered & remaining_edges
            # Must be valid: all cross pairs must be edges
            if not is_valid_biclique(L, R):
                continue
            if len(covered_remaining) > best_count:
                best_count = len(covered_remaining)
                best_biclique = (L, R, covered_remaining)
        
        if best_biclique is None:
            # Fallback: use single-edge bicliques
            for (u, v) in remaining_edges:
                bicliques.append([[int(u)], [int(v)]])
                remaining_edges.discard((u, v))
            break
        
        L, R, covered_remaining = best_biclique
        bicliques.append([sorted(L), sorted(R)])
        remaining_edges -= covered_remaining
    
    # Convert to proper types
    result = []
    for L, R in bicliques:
        L_ints = [int(x) for x in L]
        R_ints = [int(x) for x in R]
        result.append([L_ints, R_ints])
    
    return result
# EVOLVE-BLOCK-END
