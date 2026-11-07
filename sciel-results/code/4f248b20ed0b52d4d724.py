"""Biclique partition via vertex-ordered greedy expansion."""
# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _expand_biclique(L, R, adj, n):
    """Expand L and R to maximal biclique given current adj."""
    changed = True
    while changed:
        changed = False
        # Expand L
        best_l_gain = 0
        best_l_v = None
        for w in range(n):
            if w in L or w in R:
                continue
            ok = True
            for r in R:
                if r not in adj[w]:
                    ok = False
                    break
            if ok:
                gain = len(R)
                if gain > best_l_gain:
                    best_l_gain = gain
                    best_l_v = w
        if best_l_v is not None:
            L.add(best_l_v)
            changed = True
            continue
            
        # Expand R
        best_r_gain = 0
        best_r_v = None
        for w in range(n):
            if w in L or w in R:
                continue
            ok = True
            for l in L:
                if l not in adj[w]:
                    ok = False
                    break
            if ok:
                gain = len(L)
                if gain > best_r_gain:
                    best_r_gain = gain
                    best_r_v = w
        if best_r_v is not None:
            R.add(best_r_v)
            changed = True
    return L, R

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    remaining_edges = set()
    for (a, b) in P['edges']:
        adj[a].add(b)
        adj[b].add(a)
        remaining_edges.add((min(a,b), max(a,b)))
        
    bicliques = []
    
    # Process vertices in order of decreasing degree to prioritize high-connectivity nodes?
    # Or just 0 to n-1. Let's try 0 to n-1 first as a distinct baseline from "best edge".
    # Actually, processing high degree first might be better.
    vertices = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
    
    for u in vertices:
        if not adj[u]:
            continue
            
        # Find neighbors of u in remaining graph
        neighbors = list(adj[u])
        if not neighbors:
            continue
            
        # Select a starting R. 
        # Heuristic: Pick the neighbor v that has the most common neighbors with u? 
        # Or just pick the neighbor with highest degree.
        # Let's pick the neighbor v in neighbors that maximizes len(adj[v])
        best_v = max(neighbors, key=lambda x: len(adj[x]))
        
        L = {u}
        R = {best_v}
        
        # Expand to maximal biclique
        L, R = _expand_biclique(L, R, adj, n)
        
        # If expansion failed or produced trivial, ensure we have at least the edge (u, best_v)
        if u not in L or best_v not in R:
             # Fallback
             L = {u}
             R = {best_v}
        
        # Record biclique
        bicliques.append([sorted(L), sorted(R)])
        
        # Remove edges from adj and remaining_edges
        for l in L:
            for r in R:
                if l == r: continue
                e = (min(l,r), max(l,r))
                if e in remaining_edges:
                    remaining_edges.discard(e)
                    adj[l].discard(r)
                    adj[r].discard(l)
                    
    # Handle leftovers
    for e in list(remaining_edges):
        u, v = e
        bicliques.append([[u], [v]])
        
    result = []
    for (L, R) in bicliques:
        L_int = [int(x) for x in L]
        R_int = [int(x) for x in R]
        result.append([L_int, R_int])
        
    return result
# EVOLVE-BLOCK-END
