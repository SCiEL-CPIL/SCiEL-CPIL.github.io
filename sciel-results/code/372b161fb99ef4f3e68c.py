import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Normalize edges to frozensets for easy lookup
    edges_set = set()
    for u, v in P['edges']:
        edges_set.add(frozenset((u, v)))
    
    # Adjacency list for quick neighbor checks
    adj = {i: set() for i in range(P['n'])}
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        
    uncovered_edges = set(edges_set)
    bicliques = []
    
    # Helper to check if a vertex can be added to L or R
    # L, R are sets of vertices
    # We need to check connectivity in the subgraph induced by uncovered_edges
    
    while uncovered_edges:
        if time.time() > deadline:
            break
            
        # Pick an edge to start the biclique
        # Heuristic: pick an edge with high degree in the uncovered subgraph
        # For simplicity, just pick an arbitrary edge, or the first one
        # Better: pick an edge (u,v) such that u and v have high degree in uncovered graph
        
        # Calculate degrees in uncovered graph
        deg = {i: 0 for i in range(P['n'])}
        for e in uncovered_edges:
            u, v = tuple(e)
            deg[u] += 1
            deg[v] += 1
            
        # Find edge with max sum of degrees
        best_edge = None
        max_deg_sum = -1
        for e in uncovered_edges:
            u, v = tuple(e)
            s = deg[u] + deg[v]
            if s > max_deg_sum:
                max_deg_sum = s
                best_edge = e
                
        if not best_edge:
            break
            
        u, v = tuple(best_edge)
        
        # Greedy expansion to find a large biclique containing (u, v)
        L = {u}
        R = {v}
        
        # Iterate until no more vertices can be added
        changed = True
        while changed:
            changed = False
            
            # Candidates for L: vertices x not in L U R such that x is connected to all y in R via uncovered edges
            # Candidates for R: vertices x not in L U R such that x is connected to all y in L via uncovered edges
            
            # To speed up, only consider neighbors of current L or R in the uncovered graph
            # Actually, we need to check all vertices not in L U R.
            # Since N=30, this is fine.
            
            candidates_L = []
            candidates_R = []
            
            for x in range(P['n']):
                if x in L or x in R:
                    continue
                    
                # Check if x can be added to L
                # x must be connected to all y in R via uncovered edges
                can_add_L = True
                for y in R:
                    if frozenset((x, y)) not in uncovered_edges:
                        can_add_L = False
                        break
                if can_add_L:
                    # Gain: number of edges from x to L? No, x is added to L.
                    # The new edges covered are (x, y) for y in R.
                    # Wait, the biclique edges are L x R.
                    # If we add x to L, the new edges are (x, y) for all y in R.
                    # These edges must be in uncovered_edges. We already checked that.
                    # The "value" of adding x is len(R).
                    candidates_L.append((x, len(R)))
                    
                # Check if x can be added to R
                can_add_R = True
                for y in L:
                    if frozenset((x, y)) not in uncovered_edges:
                        can_add_R = False
                        break
                if can_add_R:
                    candidates_R.append((x, len(L)))
            
            # Pick the best candidate
            best_gain = 0
            best_x = None
            best_side = None
            
            if candidates_L:
                x, gain = max(candidates_L, key=lambda item: item[1])
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    best_side = 'L'
            if candidates_R:
                x, gain = max(candidates_R, key=lambda item: item[1])
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    best_side = 'R'
            
            if best_x is not None:
                if best_side == 'L':
                    L.add(best_x)
                else:
                    R.add(best_x)
                changed = True
        
        # Add the biclique
        bicliques.append([list(L), list(R)])
        
        # Remove covered edges from uncovered_edges
        for x in L:
            for y in R:
                e = frozenset((x, y))
                if e in uncovered_edges:
                    uncovered_edges.remove(e)
                    
    # Fallback: if we didn't cover all edges (e.g. due to timeout or bug),
    # cover remaining edges with single-edge bicliques
    if uncovered_edges:
        for e in uncovered_edges:
            u, v = tuple(e)
            bicliques.append([[u], [v]])
            
    # Convert to required format: list of [left, right] where left/right are lists of ints
    result = []
    for L, R in bicliques:
        # Ensure types are int
        L_int = [int(x) for x in L]
        R_int = [int(x) for x in R]
        result.append([L_int, R_int])
        
    return result
