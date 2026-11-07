import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Normalize edges to tuples (u, v) with u < v
    edges = set()
    for u, v in P['edges']:
        if u > v:
            u, v = v, u
        edges.add((u, v))
    
    uncovered = set(edges)
    partition = []
    
    # Pre-compute adjacency for initial graph? No, it changes.
    # We will compute neighborhoods on the fly from 'uncovered'.
    
    while uncovered:
        if time.time() > deadline:
            break
            
        best_biclique = None
        best_coverage = 0
        
        # Get list of vertices involved in uncovered edges
        vertices_in_uncovered = set()
        for u, v in uncovered:
            vertices_in_uncovered.add(u)
            vertices_in_uncovered.add(v)
        
        # Candidates: Stars (L={u})
        for u in vertices_in_uncovered:
            # R is neighbors of u in uncovered
            R = set()
            for v in vertices_in_uncovered:
                if v != u:
                    if (u, v) in uncovered or (v, u) in uncovered:
                        R.add(v)
            if not R:
                continue
            L = {u}
            # Coverage is len(L) * len(R) ? 
            # Actually, coverage is number of edges in L x R that are in uncovered.
            # Since R is defined as neighbors in uncovered, all (u, v) for v in R are in uncovered.
            # So coverage = len(R).
            coverage = len(R)
            if coverage > best_coverage:
                best_coverage = coverage
                best_biclique = (L, R)
                
        # Candidates: Pairs (L={u, v})
        # To keep it fast, we only check pairs that are connected in uncovered?
        # Or just all pairs in vertices_in_uncovered?
        # Number of vertices is small (<=30). Pairs ~ 435.
        # For each pair, intersection of neighborhoods.
        verts_list = list(vertices_in_uncovered)
        for i in range(len(verts_list)):
            u = verts_list[i]
            # Compute N(u) in uncovered
            Nu = set()
            for v in verts_list:
                if v != u:
                    if (u, v) in uncovered or (v, u) in uncovered:
                        Nu.add(v)
            
            for j in range(i + 1, len(verts_list)):
                v = verts_list[j]
                if (u, v) not in uncovered and (v, u) not in uncovered:
                    continue # No edge between u and v in uncovered, so (u,v) not covered by this biclique?
                    # Wait. The biclique covers L x R.
                    # If u, v in L, and R contains w, it covers (u,w) and (v,w).
                    # It does NOT cover (u,v) unless v in R or u in R.
                    # But L and R are disjoint. So (u,v) is NOT covered by this biclique.
                    # This is fine. We just need L x R subset of uncovered.
                    
                # Compute N(v) in uncovered
                Nv = set()
                for w in verts_list:
                    if w != v:
                        if (v, w) in uncovered or (w, v) in uncovered:
                            Nv.add(w)
                
                R = Nu.intersection(Nv)
                if not R:
                    continue
                L = {u, v}
                # Check if L x R is subset of uncovered
                # We know (u, w) in uncovered for w in R (since R subset Nu)
                # We know (v, w) in uncovered for w in R (since R subset Nv)
                # So yes.
                coverage = len(L) * len(R) # 2 * len(R)
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_biclique = (L, R)
                    
        if best_biclique is None:
            # Should not happen if uncovered is not empty
            # Fallback: pick any edge (u, v) and make L={u}, R={v}
            u, v = next(iter(uncovered))
            L = {u}
            R = {v}
            best_biclique = (L, R)
            
        L, R = best_biclique
        # Convert to lists of ints
        L_list = [int(x) for x in L]
        R_list = [int(x) for x in R]
        partition.append([L_list, R_list])
        
        # Remove covered edges
        for u in L:
            for v in R:
                edge = (u, v) if u < v else (v, u)
                if edge in uncovered:
                    uncovered.remove(edge)
                    
    # If we timed out, we might have uncovered edges.
    # The task requires ALL edges to be covered.
    # If we break early, the result is invalid.
    # So we must ensure we finish.
    # The greedy loop is fast, so timeout is unlikely.
    # But if it happens, we must cover the rest with single-edge bicliques.
    if uncovered:
        for u, v in uncovered:
            partition.append([[int(u)], [int(v)]])
        uncovered.clear()

    return partition

# EVOLVE-BLOCK-END
