import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = 50.0  # Safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency sets for quick lookup
    # adj[u] is a set of v such that (u,v) is an edge
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # remaining_edges is a set of (u, v) with u < v
    remaining_edges = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        remaining_edges.add((u, v))
        
    bicliques = []
    
    def find_maximal_biclique(start_u, start_v):
        """
        Finds a maximal biclique containing (start_u, start_v) using only edges in remaining_edges.
        Returns (L, R) where L and R are lists of vertices.
        """
        if (start_u, start_v) not in remaining_edges and (start_v, start_u) not in remaining_edges:
            return None
            
        # Normalize start edge
        if start_u > start_v:
            su, sv = start_v, start_u
        else:
            su, sv = start_u, start_v
            
        L = {su}
        R = {sv}
        
        changed = True
        while changed:
            changed = False
            
            # Try to add to L
            # Candidate u' must be connected to all r in R via remaining edges
            candidates_L = []
            for u_prime in range(n):
                if u_prime in L:
                    continue
                # Check if u_prime is connected to all r in R
                valid = True
                for r in R:
                    edge = (min(u_prime, r), max(u_prime, r))
                    if edge not in remaining_edges:
                        valid = False
                        break
                if valid:
                    candidates_L.append(u_prime)
                    
            if candidates_L:
                # Pick the one with highest degree in R? Or just the first?
                # Let's pick the one with max common neighbors in R to encourage growth
                # Actually, any maximal expansion is fine. Let's just pick the first to be fast.
                # Or better, pick the one that allows the most future growth?
                # For now, just pick one.
                u_add = candidates_L[0]
                L.add(u_add)
                changed = True
                
            # Try to add to R
            candidates_R = []
            for v_prime in range(n):
                if v_prime in R:
                    continue
                valid = True
                for l in L:
                    edge = (min(v_prime, l), max(v_prime, l))
                    if edge not in remaining_edges:
                        valid = False
                        break
                if valid:
                    candidates_R.append(v_prime)
                    
            if candidates_R:
                v_add = candidates_R[0]
                R.add(v_add)
                changed = True
                
        return list(L), list(R)

    while remaining_edges:
        if time.time() > deadline:
            break
            
        # Pick an edge to start with
        # Heuristic: pick edge with max sum of degrees in remaining graph?
        # For simplicity, just pick the first one.
        # To be slightly smarter, we can pick the edge that is part of the most potential bicliques.
        # Let's just pick one arbitrarily from the set.
        edge = next(iter(remaining_edges))
        u, v = edge
        
        biclique = find_maximal_biclique(u, v)
        if biclique is None:
             # Should not happen if edge is in remaining_edges
             continue
             
        L, R = biclique
        
        # Remove all edges in L x R from remaining_edges
        for l in L:
            for r in R:
                edge = (min(l, r), max(l, r))
                if edge in remaining_edges:
                    remaining_edges.remove(edge)
                    
        bicliques.append([L, R])
        
    # If we ran out of time, we might not have covered all edges.
    # The task requires partitioning ALL edges.
    # If we didn't finish, we must add the remaining edges as individual bicliques (stars or single edges) to be valid.
    if remaining_edges:
        # Fallback: add each remaining edge as a biclique of size 2 (single edge)
        # Or group them if possible quickly?
        # Single edges are always valid bicliques.
        for u, v in remaining_edges:
             bicliques.append([[u], [v]])
             
    return bicliques
# EVOLVE-BLOCK-END
