import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges_list = P['edges']
    edge_set = set()
    for e in edges_list:
        a, b = e
        if a > b:
            a, b = b, a
        edge_set.add((a, b))
    
    # Build adjacency matrix for fast lookup
    adj_matrix = [[False] * n for _ in range(n)]
    for a, b in edge_set:
        adj_matrix[a][b] = True
        adj_matrix[b][a] = True

    def get_greedy_partition(seed_val):
        random.seed(seed_val)
        remaining = set(edge_set)
        bicliques = []
        
        while remaining:
            # Pick a random edge from remaining to start a biclique
            # Or pick one with high degree to get larger bicliques
            # Let's try picking the edge with the highest sum of degrees in remaining graph
            rem_deg = [0] * n
            for a, b in remaining:
                rem_deg[a] += 1
                rem_deg[b] += 1
            
            # Sort edges by degree sum
            sorted_edges = sorted(remaining, key=lambda e: rem_deg[e[0]] + rem_deg[e[1]], reverse=True)
            
            # Pick the first one (deterministic) or a random one from top K
            # To allow local search to work better, maybe just use deterministic first
            # But to get diverse starting points, we can shuffle the top ones
            top_k = min(5, len(sorted_edges))
            seed_edge = random.choice(sorted_edges[:top_k])
            
            u, v = seed_edge
            L = {u}
            R = {v}
            
            # Expand to maximal biclique
            changed = True
            while changed:
                changed = False
                # Add to L
                candidates_L = [x for x in range(n) if x not in L and x not in R]
                for x in candidates_L:
                    if all(adj_matrix[x][y] for y in R):
                        # Check if adding x doesn't create edges not in remaining?
                        # No, we just need it to be a biclique in the original graph.
                        # But we must ensure we only cover edges in 'remaining'.
                        # If x connects to y in R, and (x,y) is in remaining, it's fine.
                        # If (x,y) is NOT in remaining, it means it was already covered by another biclique.
                        # In that case, we CANNOT add x to L if it connects to y in R via an already covered edge?
                        # Wait. The biclique covers ALL edges between L and R.
                        # If we add x to L, the biclique will cover (x,y) for all y in R.
                        # If (x,y) is not in 'remaining', it is already covered.
                        # So we would be covering it twice.
                        # Therefore, we can only add x if ALL (x,y) for y in R are in 'remaining'.
                        if all((min(x, y), max(x, y)) in remaining for y in R):
                            L.add(x)
                            changed = True
                            
                # Add to R
                candidates_R = [y for y in range(n) if y not in L and y not in R]
                for y in candidates_R:
                    if all((min(x, y), max(x, y)) in remaining for x in L):
                        R.add(y)
                        changed = True
            
            # Record biclique
            if L and R:
                bicliques.append((L, R))
                # Remove edges
                for l in L:
                    for r in R:
                        e = (min(l, r), max(l, r))
                        remaining.discard(e)
            
            if not remaining:
                break
                
        return bicliques

    def local_search(bicliques):
        # Convert to list of [L, R]
        # And track edges covered
        covered_edges = set()
        biclique_edges = []
        for L, R in bicliques:
            edges = set()
            for l in L:
                for r in R:
                    e = (min(l, r), max(l, r))
                    edges.add(e)
            biclique_edges.append(edges)
            covered_edges.update(edges)
            
        improved = True
        while improved:
            improved = False
            n_b = len(bicliques)
            for i in range(n_b):
                for j in range(i + 1, n_b):
                    L1, R1 = bicliques[i]
                    L2, R2 = bicliques[j]
                    
                    L_new = L1 | L2
                    R_new = R1 | R2
                    
                    # Check disjointness
                    if L_new & R_new:
                        continue
                    
                    # Check if union is a valid biclique in the graph
                    # And if it covers exactly E1 U E2
                    E1 = biclique_edges[i]
                    E2 = biclique_edges[j]
                    E_target = E1 | E2
                    
                    # Check all pairs in L_new x R_new are in E_target
                    valid = True
                    for l in L_new:
                        for r in R_new:
                            e = (min(l, r), max(l, r))
                            if e not in E_target:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        # Merge
                        bicliques[i] = (L_new, R_new)
                        biclique_edges[i] = E_target
                        bicliques.pop(j)
                        biclique_edges.pop(j)
                        improved = True
                        break
                if improved:
                    break
                    
        return bicliques

    # Run multiple times with different seeds to find best initial
    best_bicliques = None
    best_count = float('inf')
    
    start_time = time.time()
    deadline = start_time + 80.0
    
    # Try a few random seeds for initialization
    for seed in range(10):
        if time.time() > deadline:
            break
        bics = get_greedy_partition(seed)
        if len(bics) < best_count:
            best_count = len(bics)
            best_bicliques = bics
            
    # Local search on the best one
    if best_bicliques:
        final_bics = local_search(best_bicliques)
        best_count = len(final_bics)
        best_bicliques = final_bics
        
    # If we still have time, maybe try local search on other initializations?
    # But merging is monotonic (reduces count), so local search on a worse start might not beat the best start's local search.
    # However, a worse start might have different structure.
    # Given time, stick with best.
    
    # Convert to required format
    result = []
    for L, R in best_bicliques:
        left = [int(x) for x in L]
        right = [int(x) for x in R]
        result.append([left, right])
        
    return result
