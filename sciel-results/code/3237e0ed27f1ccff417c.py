"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    deadline = time.time() + 80.0
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Edge set for O(1) lookup
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u,v), max(u,v)))
    
    def grow_biclique(start_u, start_v, remaining):
        """Grow a biclique from edge (start_u, start_v)."""
        if (min(start_u,start_v), max(start_u,start_v)) not in remaining:
            return None
        L = [start_u]
        R = [start_v]
        
        # Iteratively add vertices
        changed = True
        while changed:
            if time.time() > deadline:
                return None
            changed = False
            # Try adding to L
            for x in range(n):
                if x in L or x in R:
                    continue
                # x must be connected to all in R via remaining edges
                ok = True
                for r in R:
                    if (min(x,r), max(x,r)) not in remaining:
                        ok = False
                        break
                if ok:
                    L.append(x)
                    changed = True
            # Try adding to R
            for y in range(n):
                if y in L or y in R:
                    continue
                ok = True
                for l in L:
                    if (min(l,y), max(l,y)) not in remaining:
                        ok = False
                        break
                if ok:
                    R.append(y)
                    changed = True
        
        # Verify
        for l in L:
            for r in R:
                if (min(l,r), max(l,r)) not in edge_set:
                    return None
        return (L, R)
    
    def greedy_partition(remaining, adj, edge_set):
        """Greedy biclique extraction."""
        bicliques = []
        rem = set(remaining)
        
        while rem:
            if time.time() > deadline:
                break
            # Pick an edge - try to find one that leads to large biclique
            # Use heuristic: pick edge with high-degree endpoints
            edge_list = list(rem)
            
            # Score each edge by degree product
            best_edge = None
            best_score = -1
            for (u, v) in edge_list[:min(30, len(edge_list))]:
                score = len(adj[u]) * len(adj[v])
                if score > best_score:
                    best_score = score
                    best_edge = (u, v)
            
            if best_edge is None:
                best_edge = edge_list[0]
            
            u, v = best_edge
            bic = grow_biclique(u, v, rem)
            
            if bic is None:
                bic = ([u], [v])
            
            L, R = bic
            bicliques.append((L, R))
            
            # Remove covered edges
            for l in L:
                for r in R:
                    rem.discard((min(l,r), max(l,r)))
        
        return bicliques
    
    best_result = None
    best_count = float('inf')
    
    # Try multiple strategies
    for attempt in range(30):
        if time.time() > deadline:
            break
        
        rem = set(edge_set)
        
        if attempt == 0:
            # Standard greedy
            bicliques = greedy_partition(rem, adj, edge_set)
        else:
            # Randomized greedy: shuffle edge selection
            random.seed(attempt)
            bicliques = []
            r = set(rem)
            while r:
                if time.time() > deadline:
                    break
                edge_list = list(r)
                random.shuffle(edge_list)
                u, v = edge_list[0]
                bic = grow_biclique(u, v, r)
                if bic is None:
                    bic = ([u], [v])
                L, R = bic
                bicliques.append((L, R))
                for l in L:
                    for r_ in R:
                        r.discard((min(l,r_), max(l,r_)))
        
        if len(bicliques) < best_count:
            best_count = len(bicliques)
            best_result = bicliques
        
        if best_count <= 10:
            break
    
    if best_result is None:
        # Fallback
        best_result = [([u], [v for a,v in edges if a==u]) for u in range(n) if any(a==u for a,v in edges)]
    
    result = []
    for L, R in best_result:
        result.append(([int(x) for x in L], [int(x) for x in R]))
    
    return result
# EVOLVE-BLOCK-END
