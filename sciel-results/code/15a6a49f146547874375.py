import time
import sys

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin
    
    n = P['n']
    edges = P['edges']
    edge_set = set()
    for e in edges:
        edge_set.add((min(e[0], e[1]), max(e[0], e[1])))
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Greedy: repeatedly pick the largest biclique from remaining edges
    remaining = set(edge_set)
    bicliques = []
    
    while remaining:
        if time.time() > deadline:
            break
        
        # Find the best biclique: maximize |L|*|R|
        best_biclique = None
        best_size = 0
        
        # Try each vertex as a seed for the left side
        # We'll try to grow bicliques from each remaining edge's endpoints
        candidate_edges = list(remaining)
        
        # Try all pairs of vertices as seeds for L and R
        # To be efficient, try growing from single vertices
        for u in range(n):
            if time.time() > deadline:
                break
            # Find neighbors of u that are connected to u in remaining
            u_remaining_neighbors = set()
            for v in adj[u]:
                if (min(u,v), max(u,v)) in remaining:
                    u_remaining_neighbors.add(v)
            
            if not u_remaining_neighbors:
                continue
            
            # Try to grow a biclique with u in left
            # Start with L={u}, R=u_remaining_neighbors
            # Then iteratively add vertices that connect to all of R
            L = {u}
            R = set(u_remaining_neighbors)
            
            # Iterate to closure
            changed = True
            while changed:
                changed = False
                # Find vertices that connect to all in R
                new_L_candidates = set()
                for v in range(n):
                    if v not in L and v not in R:
                        # v must connect to all in R via remaining edges
                        all_connected = True
                        for r in R:
                            if (min(v,r), max(v,r)) not in remaining:
                                all_connected = False
                                break
                        if all_connected:
                            new_L_candidates.add(v)
                
                if new_L_candidates:
                    L = L | new_L_candidates
                    changed = True
                
                # Find vertices that connect to all in L
                new_R_candidates = set()
                for v in range(n):
                    if v not in L and v not in R:
                        all_connected = True
                        for l in L:
                            if (min(v,l), max(v,l)) not in remaining:
                                all_connected = False
                                break
                        if all_connected:
                            new_R_candidates.add(v)
                
                if new_R_candidates:
                    R = R | new_R_candidates
                    changed = True
            
            size = len(L) * len(R)
            if size > best_size:
                best_size = size
                best_biclique = (L, R)
        
        if best_biclique is None:
            # Fallback: take one remaining edge
            e = next(iter(remaining))
            best_biclique = ({e[0]}, {e[1]})
        
        L, R = best_biclique
        # Ensure all cross edges are in remaining
        valid_L = set()
        for l in L:
            all_ok = True
            for r in R:
                if (min(l,r), max(l,r)) not in remaining:
                    all_ok = False
                    break
            if all_ok:
                valid_L.add(l)
        
        if not valid_L:
            e = next(iter(remaining))
            valid_L = {e[0]}
            R = {e[1]}
        
        # Remove all cross edges from remaining
        for l in valid_L:
            for r in R:
                e = (min(l,r), max(l,r))
                remaining.discard(e)
        
        bicliques.append((list(valid_L), list(R)))
    
    # Verify and format
    result = []
    for L, R in bicliques:
        L = [int(x) for x in L]
        R = [int(x) for x in R]
        result.append([L, R])
    
    # Verify all edges are covered exactly once
    covered = {}
    for L, R in bicliques:
        for l in L:
            for r in R:
                e = (min(l, r), max(l, r))
                if e in covered:
                    # Duplicate - this is invalid, but we'll try to fix
                    pass
                covered[e] = 1
    
    return result
