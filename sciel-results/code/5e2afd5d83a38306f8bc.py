"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency structure
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u,v), max(u,v)))
    
    remaining_edges = set(edge_set)
    bicliques = []
    
    # Helper to check if a biclique (L, R) is valid (all cross edges exist)
    # and returns the set of edges it covers that are in `remaining`.
    def evaluate_biclique(L, R, remaining):
        if not L or not R:
            return set()
        # Check validity: all u in L, v in R must be edges in original graph
        for u in L:
            for v in R:
                if v not in adj[u]:
                    return set()
        # Calculate covered edges in remaining
        covered = set()
        for u in L:
            for v in R:
                e = (min(u,v), max(u,v))
                if e in remaining:
                    covered.add(e)
        return covered

    # Function to expand a seed biclique to a maximal one with respect to edge_set
    # We use a heuristic expansion: repeatedly add vertices that connect to all current members of the opposite side.
    def expand_to_maximal(L, R):
        L = set(L)
        R = set(R)
        changed = True
        while changed:
            changed = False
            # Try adding to L
            candidates_L = set()
            for w in range(n):
                if w in L or w in R:
                    continue
                # w must be connected to all in R
                if all(v in adj[w] for v in R):
                    candidates_L.add(w)
            
            # Add candidates that actually cover more edges? 
            # For maximal biclique, we just need validity. 
            # However, to keep it "useful", we might restrict to those covering remaining edges.
            # But the prompt says "maximal bipartite subgraph in the remaining edge set".
            # This implies we should look at the subgraph induced by remaining edges?
            # No, "biclique partition of a random graph... Every cross pair must be a host edge."
            # The biclique must be valid in the HOST graph.
            # The "maximal in remaining" likely refers to the coverage objective.
            
            # Let's just expand to maximal valid biclique in host graph.
            # We add all possible candidates to ensure maximality.
            # But adding too many might dilute the coverage of *remaining* edges if we don't care about already covered ones?
            # Actually, if we add a vertex that doesn't cover any *remaining* edges, it doesn't help.
            # So we only add if it covers at least one remaining edge.
            
            # To be efficient, let's just try to add one by one or all at once.
            # Adding all at once is faster but might add vertices that don't help.
            # Let's check if adding w to L covers any remaining edge.
            new_L = set()
            for w in candidates_L:
                # Does w cover any remaining edge with R?
                has_cover = False
                for v in R:
                    e = (min(w,v), max(w,v))
                    if e in remaining_edges:
                        has_cover = True
                        break
                if has_cover:
                    new_L.add(w)
            
            if new_L:
                L |= new_L
                changed = True
            
            # Try adding to R
            candidates_R = set()
            for w in range(n):
                if w in L or w in R:
                    continue
                if all(u in adj[w] for u in L):
                    candidates_R.add(w)
            
            new_R = set()
            for w in candidates_R:
                has_cover = False
                for u in L:
                    e = (min(w,u), max(w,u))
                    if e in remaining_edges:
                        has_cover = True
                        break
                if has_cover:
                    new_R.add(w)
            
            if new_R:
                R |= new_R
                changed = True
        
        return L, R

    # Generate a pool of candidate bicliques
    # Strategy:
    # 1. For each edge (u,v), try expanding ({u}, {v})
    # 2. For each vertex u, try expanding ({u}, N(u)) - this might be too big, so maybe ({u}, subset)
    #    Better: For each vertex u, pick a few neighbors and expand.
    # 3. Random pairs?
    
    candidates = []
    
    # 1. Edge-based seeds
    for u, v in edges:
        L, R = expand_to_maximal({u}, {v})
        if L and R:
            # Store as frozensets for uniqueness
            key = (frozenset(L), frozenset(R))
            if key not in candidates:
                candidates.append((L, R))
        
        # Also try swapping L and R? No, biclique is unordered pair of sets, but (L,R) and (R,L) are same structure.
        # However, the expansion depends on the seed. ({u},{v}) vs ({v},{u}) might yield different maximal bicliques?
        # Yes, because the "cover remaining" check is asymmetric in the iteration if we don't handle it carefully.
        # But expand_to_maximal adds to L if it covers remaining with R, and to R if it covers with L.
        # It should converge to the same maximal biclique if we keep iterating.
        # Let's just add the result.
        
    # 2. Vertex-neighborhood seeds
    # For each vertex, try a biclique with L={u} and R being a random subset of its neighbors?
    # Or L={u} and R=N(u)?
    # N(u) might be large. Let's try L={u}, R=N(u) and expand.
    for u in range(n):
        if not adj[u]:
            continue
        R_seed = set(adj[u])
        # If R_seed is too large, it might be inefficient to check validity for all.
        # But n=30, so it's fine.
        L, R = expand_to_maximal({u}, R_seed)
        if L and R:
            key = (frozenset(L), frozenset(R))
            if key not in candidates:
                candidates.append((L, R))
        
        # Also try L=N(u), R={u}
        L_seed = set(adj[u])
        L, R = expand_to_maximal(L_seed, {u})
        if L and R:
            key = (frozenset(L), frozenset(R))
            if key not in candidates:
                candidates.append((L, R))

    # 3. Pair of neighbors seeds
    # For each u, pick two neighbors v, w. Try L={u}, R={v,w}.
    # This might find bicliques where u is on one side and multiple neighbors on the other.
    for u in range(n):
        neighbors = list(adj[u])
        if len(neighbors) < 2:
            continue
        # Sample a few pairs to avoid O(n^3) explosion
        import random
        random.seed(42) # Deterministic for reproducibility in this block? 
        # Actually, we want deterministic output. 
        # Let's just iterate all pairs if n is small. n=30, neighbors ~10. Pairs ~45. 30*45=1350 seeds. Fine.
        for i in range(len(neighbors)):
            for j in range(i+1, len(neighbors)):
                v = neighbors[i]
                w = neighbors[j]
                # Check if v-w is an edge? Not required for L={u}, R={v,w} to be a valid biclique seed.
                # Wait, L={u}, R={v,w} requires u-v and u-w to be edges. They are.
                # It does NOT require v-w to be an edge.
                L, R = expand_to_maximal({u}, {v, w})
                if L and R:
                    key = (frozenset(L), frozenset(R))
                    if key not in candidates:
                        candidates.append((L, R))

    # Greedy Set Cover
    # We have a list of candidates (L, R).
    # At each step, pick the one covering the most remaining_edges.
    
    # To speed up, we can pre-compute the full edge set for each candidate?
    # But we need intersection with remaining_edges.
    
    while remaining_edges:
        best_biclique = None
        best_covered = set()
        best_count = 0
        
        for L, R in candidates:
            # Quick check: if L or R is empty, skip
            if not L or not R:
                continue
            
            # Calculate intersection with remaining_edges
            # Optimization: if the candidate's total edges are fewer than best_count, skip?
            # We don't have total edges pre-computed.
            
            covered = set()
            # Iterate through L and R to find edges in remaining
            # This is O(|L|*|R|).
            # If candidates are large, this is slow.
            # But n=30, so max 30*30=900 checks per candidate.
            # If we have ~1000 candidates, that's 900,000 checks per iteration.
            # If we need ~20 iterations, that's 18M checks. Python might be slow (1-2 seconds).
            # Acceptable within 60s.
            
            for u in L:
                for v in R:
                    e = (min(u,v), max(u,v))
                    if e in remaining_edges:
                        covered.add(e)
            
            count = len(covered)
            if count > best_count:
                best_count = count
                best_covered = covered
                best_biclique = (L, R)
        
        if best_biclique is None or best_count == 0:
            # Fallback: single edges
            for e in remaining_edges:
                u, v = e
                bicliques.append([[int(u)], [int(v)]])
            break
        
        L, R = best_biclique
        bicliques.append([sorted(L), sorted(R)])
        remaining_edges -= best_covered
        
        # Optional: Remove candidates that are subsets of the chosen one?
        # Or just let them be re-evaluated (they will cover 0 new edges).
        
    # Convert to proper types
    result = []
    for L, R in bicliques:
        L_ints = [int(x) for x in L]
        R_ints = [int(x) for x in R]
        result.append([L_ints, R_ints])
    
    return result
# EVOLVE-BLOCK-END
