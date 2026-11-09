"""Biclique partitioning of a random graph using greedy set-cover with local search."""
import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    n = P['n']
    edges = P['edges']
    
    # Build adjacency structure
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Total edges to cover
    all_edges = set()
    for u, v in edges:
        if u < v:
            all_edges.add((u, v))
        else:
            all_edges.add((v, u))
    
    total_edges = len(all_edges)
    
    def find_max_biclique(remaining_edges):
        """Find a large biclique from remaining edges."""
        if not remaining_edges:
            return None, None
        
        # Try to find a large biclique by looking for dense regions
        # Start with a high-degree vertex
        deg = {}
        for u, v in remaining_edges:
            deg[u] = deg.get(u, 0) + 1
            deg[v] = deg.get(v, 0) + 1
        
        if not deg:
            return None, None
        
        # Pick a vertex with max degree
        start = max(deg, key=lambda x: deg[x])
        
        # Build candidate left and right sets
        # A biclique: all edges between L and R must be in remaining_edges
        # Strategy: pick a seed vertex, find its neighbors, then refine
        
        # Get all neighbors of start in remaining edges
        neighbors = set()
        for u, v in remaining_edges:
            if u == start:
                neighbors.add(v)
            elif v == start:
                neighbors.add(u)
        
        if not neighbors:
            return None, None
        
        # Try different orientations: start on left or right
        best_biclique = None
        best_size = 0
        
        for orientation in [0, 1]:
            if orientation == 0:
                L = {start}
                R = set(neighbors)
            else:
                L = set(neighbors)
                R = {start}
            
            # Iteratively refine: keep only vertices that connect to all on the other side
            changed = True
            while changed:
                changed = False
                # For each v in R, check if it connects to all in L
                new_R = set()
                for v in R:
                    ok = True
                    for u in L:
                        e = (min(u, v), max(u, v))
                        if e not in remaining_edges:
                            ok = False
                            break
                    if ok:
                        new_R.add(v)
                    else:
                        changed = True
                R = new_R
                
                # For each u in L, check if it connects to all in R
                new_L = set()
                for u in L:
                    ok = True
                    for v in R:
                        e = (min(u, v), max(u, v))
                        if e not in remaining_edges:
                            ok = False
                            break
                    if ok:
                        new_L.add(u)
                    else:
                        changed = True
                L = new_L
            
            if L and R:
                size = len(L) * len(R)
                if size > best_size:
                    best_size = size
                    best_biclique = (L, R)
        
        if best_biclique is None:
            # Fallback: just a star
            L = {start}
            R = set(neighbors)
            return L, R
        
        return best_biclique
    
    def greedy_partition():
        """Greedy set-cover style partition."""
        remaining = set(all_edges)
        bicliques = []
        
        while remaining:
            if time.time() > deadline:
                break
            
            L, R = find_max_biclique(remaining)
            if L is None or not L or not R:
                # Fallback: take one edge
                u, v = next(iter(remaining))
                L = {u}
                R = {v}
            
            # Remove all edges between L and R from remaining
            for u in L:
                for v in R:
                    e = (min(u, v), max(u, v))
                    remaining.discard(e)
            
            bicliques.append((list(L), list(R)))
        
        return bicliques, remaining
    
    def try_local_improvement(bicliques):
        """Try to merge or improve bicliques."""
        # Simple local search: try to see if we can reassign edges
        # This is a placeholder for more sophisticated local search
        return bicliques
    
    # Run greedy
    bicliques, remaining = greedy_partition()
    
    # If there are remaining edges (due to deadline), add them as single-edge bicliques
    if remaining:
        for u, v in remaining:
            bicliques.append(([u], [v]))
    
    # Try local improvement if time permits
    if time.time() < deadline - 5.0 and len(bicliques) > 1:
        bicliques = try_local_improvement(bicliques)
    
    # Convert to required format: list of [left_vertices, right_vertices]
    result = []
    for L, R in bicliques:
        result.append([list(L), list(R)])
    
    return result
