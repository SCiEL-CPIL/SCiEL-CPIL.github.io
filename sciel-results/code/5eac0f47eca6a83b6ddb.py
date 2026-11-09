import random
import time

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Precompute adjacency matrix
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        
    # Helper to check if adding edge (u,v) to biclique (L,R) is possible
    # L, R are sets.
    # We want to form (L', R') such that (L', R') is a biclique and covers (L,R) plus (u,v).
    # Actually, we just check if (u,v) can be part of a biclique that contains L and R.
    # This is equivalent to:
    # 1. If u in L and v in R: Already there (but we process each edge once, so this won't happen in online).
    # 2. If u in L and v not in R: Can we add v to R?
    #    Check: for all l in L, adj[l][v] is True.
    # 3. If v in R and u not in L: Can we add u to L?
    #    Check: for all r in R, adj[u][r] is True.
    # 4. If u not in L and v not in R: Can we add both?
    #    Check: for all l in L, adj[l][v] is True.
    #    Check: for all r in R, adj[u][r] is True.
    #    (adj[u][v] is True by definition of edge).
    
    def can_add(u, v, L, R):
        # Returns (new_L, new_R) if possible, else None
        if u in L and v in R:
            return (L, R) # Should not happen in online construction of new edges
        
        new_L = set(L)
        new_R = set(R)
        
        if u not in L:
            new_L.add(u)
        if v not in R:
            new_R.add(v)
        
        # Check all cross pairs in new_L x new_R
        for lu in new_L:
            for rv in new_R:
                if not adj[lu][rv]:
                    return None
        return (new_L, new_R)

    def online_greedy(edge_order):
        bicliques = [] # List of (L_set, R_set)
        for u, v in edge_order:
            best_idx = -1
            best_new_biclique = None
            # Try to fit into existing bicliques
            # We prefer the one that results in the smallest increase in "complexity" or just the first one?
            # Let's try to fit into the one that allows the largest resulting biclique?
            # Or just the first one?
            # To be deterministic-ish but varied, we can randomize the order of bicliques we check.
            indices = list(range(len(bicliques)))
            random.shuffle(indices)
            
            for idx in indices:
                L, R = bicliques[idx]
                res = can_add(u, v, L, R)
                if res is not None:
                    # Accept the first one we find? Or the best?
                    # Let's accept the first one to keep it fast.
                    # Or pick the one with max size?
                    # Let's pick the one that results in the largest biclique (max |L|*|R|)
                    # This might be expensive if we check all.
                    # Given small n, let's check all and pick best.
                    if best_idx == -1:
                        best_idx = idx
                        best_new_biclique = res
                    else:
                        # Compare sizes
                        old_size = len(best_new_biclique[0]) * len(best_new_biclique[1])
                        new_size = len(res[0]) * len(res[1])
                        if new_size > old_size:
                            best_idx = idx
                            best_new_biclique = res
            
            if best_idx != -1:
                bicliques[best_idx] = best_new_biclique
            else:
                bicliques.append(({u}, {v}))
        
        return bicliques

    def local_search(bicliques):
        # Try to move edges between bicliques to reduce count
        # This is complex. 
        # Simpler: Try to merge two bicliques if their union is a biclique.
        # Or: Take edges from two bicliques, clear them, and re-greedy them.
        
        improved = True
        while improved:
            improved = False
            # Try to merge pairs
            for i in range(len(bicliques)):
                for j in range(i+1, len(bicliques)):
                    L1, R1 = bicliques[i]
                    L2, R2 = bicliques[j]
                    # Check if L1 U L2 and R1 U R2 form a biclique?
                    # No, that's too strong.
                    # Check if we can re-partition the edges of B_i and B_j into fewer bicliques.
                    # Let's just try to merge if (L1 U L2, R1 U R2) is a biclique.
                    # This is rare.
                    
                    # Better: Try to move an edge from B_i to B_j.
                    # Pick a random edge from B_i.
                    # If it fits in B_j, move it.
                    # If B_i becomes empty, remove it.
                    
                    edges_in_i = [(u, v) for u in L1 for v in R1 if adj[u][v]]
                    if not edges_in_i:
                        continue
                    random.shuffle(edges_in_i)
                    
                    for u, v in edges_in_i[:5]: # Check a few
                        res = can_add(u, v, L2, R2)
                        if res is not None:
                            # Move (u,v) to B_j
                            # Remove (u,v) from B_i?
                            # B_i is defined by L1, R1. Removing an edge doesn't change L1, R1 unless we shrink them.
                            # The biclique B_i covers ALL cross edges in L1 x R1.
                            # If we "move" an edge, we must shrink B_i or B_j.
                            # This logic is flawed for "move edge" in this representation.
                            # The representation (L,R) implies covering L x R.
                            # So we cannot just "move" one edge. We must re-partition the set of edges covered by B_i and B_j.
                            
                            # Let's collect all edges covered by B_i and B_j.
                            edges_to_repartition = []
                            for u in L1:
                                for v in R1:
                                    if adj[u][v]:
                                        edges_to_repartition.append((u, v))
                            for u in L2:
                                for v in R2:
                                    if adj[u][v]:
                                        edges_to_repartition.append((u, v))
                            
                            # Repartition these edges using online greedy?
                            # This might increase the count.
                            # We only accept if it decreases the count.
                            # But online greedy is not optimal.
                            # Let's use a simple greedy:
                            # 1. Find largest biclique in this set.
                            # 2. Remove it.
                            # 3. Repeat.
                            
                            # This is expensive.
                            pass
            # Skip complex local search for now, rely on multiple restarts of online greedy.
        return bicliques

    best_solution = None
    best_count = float('inf')
    
    start_time = time.time()
    max_time = 55.0
    
    # Generate multiple random orderings
    while time.time() - start_time < max_time:
        # Random ordering
        edge_order = list(edges)
        random.shuffle(edge_order)
        
        sol = online_greedy(edge_order)
        count = len(sol)
        
        if count < best_count:
            best_count = count
            best_solution = sol
            
        # Also try degree-sorted ordering
        # Calculate degrees
        degrees = [0] * n
        for u, v in edges:
            degrees[u] += 1
            degrees[v] += 1
            
        edge_order_deg = sorted(edges, key=lambda e: degrees[e[0]] + degrees[e[1]], reverse=True)
        sol_deg = online_greedy(edge_order_deg)
        count_deg = len(sol_deg)
        
        if count_deg < best_count:
            best_count = count_deg
            best_solution = sol_deg

    if best_solution is None:
        # Fallback: single edge bicliques
        best_solution = [({u}, {v}) for u, v in edges]
        
    result = []
    for L, R in best_solution:
        L_list = [int(x) for x in sorted(L)]
        R_list = [int(x) for x in sorted(R)]
        result.append([L_list, R_list])
        
    return result
