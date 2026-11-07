import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _edge_key(u, v):
    if u > v:
        u, v = v, u
    return (u, v)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = P['n']
    edges = P['edges']
    
    # Original graph adjacency sets
    orig_adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        orig_adj[u].add(v)
        orig_adj[v].add(u)
        edge_set.add(_edge_key(u, v))
        
    def greedy_cover(seed):
        random.seed(seed)
        remaining_edges = set(edge_set)
        # rem_adj[u] is set of neighbors v such that (u,v) is in remaining_edges
        rem_adj = [set() for _ in range(n)]
        for u, v in remaining_edges:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
            
        bicliques = []
        
        while remaining_edges:
            if time.time() > deadline - 5.0:
                break
                
            # Find best biclique in remaining graph
            best_L = None
            best_R = None
            best_size = -1
            
            # Sample a subset of edges to start from to save time, or all if small
            # With n=30, edges ~150. Iterating all is fine.
            # To add randomness, shuffle the list of edges
            candidates = list(remaining_edges)
            random.shuffle(candidates)
            
            for u, v in candidates:
                # Expand (u, v) in remaining graph
                L = {u}
                R = {v}
                changed = True
                while changed:
                    changed = False
                    # Try adding to L
                    # w can join L if w is connected to all in R in remaining graph
                    for w in rem_adj[u]: # Only check neighbors of u? No, w must be connected to ALL in R.
                        # Optimization: w must be in intersection of neighbors of all x in R.
                        # But iterating all w in range(n) is small.
                        pass
                    
                    # Efficient expansion
                    # Candidates for L: vertices connected to all in R
                    cand_L = set(range(n))
                    for x in R:
                        cand_L &= rem_adj[x]
                    cand_L -= L
                    cand_L -= R
                    
                    if cand_L:
                        # Pick one? Or add all?
                        # Adding all at once is faster for maximal biclique
                        L |= cand_L
                        changed = True
                        
                    # Candidates for R: vertices connected to all in L
                    cand_R = set(range(n))
                    for x in L:
                        cand_R &= rem_adj[x]
                    cand_R -= L
                    cand_R -= R
                    
                    if cand_R:
                        R |= cand_R
                        changed = True
                        
                size = len(L) * len(R)
                if size > best_size:
                    best_size = size
                    best_L = L
                    best_R = R
                    
            if best_L is None:
                # Fallback: take first edge
                u, v = next(iter(remaining_edges))
                bicliques.append((frozenset({u}), frozenset({v})))
                e = _edge_key(u, v)
                remaining_edges.remove(e)
                rem_adj[u].remove(v)
                rem_adj[v].remove(u)
            else:
                bicliques.append((frozenset(best_L), frozenset(best_R)))
                # Remove edges
                for u_l in best_L:
                    for v_r in best_R:
                        e = _edge_key(u_l, v_r)
                        if e in remaining_edges:
                            remaining_edges.remove(e)
                            rem_adj[u_l].remove(v_r)
                            rem_adj[v_r].remove(u_l)
                            
        return bicliques

    def merge_pass(bicliques):
        # Convert to mutable lists
        bicliques = [(list(L), list(R)) for L, R in bicliques]
        merged = True
        while merged and time.time() < deadline - 2.0:
            merged = False
            for i in range(len(bicliques)):
                if time.time() > deadline - 2.0:
                    break
                for j in range(i + 1, len(bicliques)):
                    if time.time() > deadline - 2.0:
                        break
                    Li, Ri = bicliques[i]
                    Lj, Rj = bicliques[j]
                    
                    # Check if no edges between Li and Rj, and Lj and Ri in ORIGINAL graph
                    valid = True
                    for u in Li:
                        for v in Rj:
                            if v in orig_adj[u]:
                                valid = False
                                break
                        if not valid: break
                    if valid:
                        for u in Lj:
                            for v in Ri:
                                if v in orig_adj[u]:
                                    valid = False
                                    break
                            if not valid: break
                    
                    if valid:
                        # Merge
                        new_L = set(Li) | set(Lj)
                        new_R = set(Ri) | set(Rj)
                        bicliques[i] = (list(new_L), list(new_R))
                        bicliques.pop(j)
                        merged = True
                        break
                if merged:
                    break
        return bicliques

    best_bicliques = None
    best_count = float('inf')
    
    # Run multiple restarts
    num_restarts = 100
    for s in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        b = greedy_cover(s)
        b = merge_pass(b)
        if len(b) < best_count:
            best_count = len(b)
            best_bicliques = b
            
    if best_bicliques is None:
        # Fallback
        return [[ [u], [v] ] for u, v in edges]
        
    # Final validation
    covered = set()
    valid = True
    for L, R in best_bicliques:
        if not L or not R:
            valid = False
            break
        if set(L) & set(R):
            valid = False
            break
        for u in L:
            for v in R:
                k = _edge_key(u, v)
                if k not in edge_set:
                    valid = False
                    break
                if k in covered:
                    valid = False
                    break
                covered.add(k)
                
    if not valid or covered != edge_set:
        return [[ [u], [v] ] for u, v in edges]
        
    # Format output
    result = []
    for L, R in best_bicliques:
        L_int = [int(x) for x in L]
        R_int = [int(x) for x in R]
        result.append([L_int, R_int])
        
    return result

# EVOLVE-BLOCK-END
