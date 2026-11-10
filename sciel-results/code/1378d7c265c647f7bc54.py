def construct():
    from random import Random
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)
    
    def solve(n, k, p, seed):
        edges = instance(n, k, p, seed)
        C = list(range(k))
        I = list(range(k, n))
        
        # cross[i] = list of c in C such that (i,c) is an edge
        cross = {i: [] for i in I}
        all_C_edges = set(frozenset((u, v)) for u in range(k) for v in range(u+1, k))
        
        for (a, b) in edges:
            if a < k and b < k:
                pass
            elif a >= k and b >= k:
                pass
            else:
                if a < k:
                    c, i = a, b
                else:
                    c, i = b, a
                cross[i].append(c)
                
        # State for greedy packing
        uncovered = {i: set(cross[i]) for i in I}
        avail_C = set(all_C_edges)
        
        mixed_cliques = []
        
        # Greedy loop
        while any(uncovered[i] for i in I):
            best_clique = None
            best_count = 0
            best_i = None
            best_S = None
            
            for i in I:
                if not uncovered[i]:
                    continue
                neighbors = uncovered[i]
                # Find a greedy maximal clique S in neighbors using avail_C
                # Sort neighbors by degree in avail_C restricted to neighbors?
                # Just iterate and try to add
                S = []
                S_set = set()
                # To improve quality, sort candidates by degree in avail_C
                # Degree of c in avail_C within neighbors
                def deg(c):
                    cnt = 0
                    for other in neighbors:
                        if other != c and frozenset((c, other)) in avail_C:
                            cnt += 1
                    return cnt
                
                sorted_neighbors = sorted(neighbors, key=deg, reverse=True)
                
                for c in sorted_neighbors:
                    # Check if c can be added to S
                    # All edges (c, s) for s in S must be in avail_C
                    can_add = True
                    for s in S:
                        if frozenset((c, s)) not in avail_C:
                            can_add = False
                            break
                    if can_add:
                        S.append(c)
                        S_set.add(c)
                        
                if len(S) > best_count:
                    best_count = len(S)
                    best_i = i
                    best_S = S
                    best_clique = [i] + S
            
            if best_count == 0:
                # Should not happen if uncovered is not empty, 
                # because we can always pick a single c (S of size 1)
                # which requires no internal edges.
                # Unless uncovered[i] is empty?
                # If best_count is 0, it means S was empty for all i with uncovered.
                # This implies neighbors was empty?
                # But we check `if not uncovered[i]: continue`.
                # So neighbors is not empty.
                # So S should have at least 1 element.
                # Why would S be empty?
                # `for c in sorted_neighbors`: if sorted_neighbors is not empty, loop runs.
                # `can_add` is True for first c (S is empty).
                # So S will have at least 1 element.
                # So best_count >= 1.
                break
            
            mixed_cliques.append(best_clique)
            
            # Update uncovered
            uncovered[best_i] -= set(best_S)
            
            # Update avail_C: remove internal edges of best_S
            for idx_a in range(len(best_S)):
                for idx_b in range(idx_a + 1, len(best_S)):
                    e = frozenset((best_S[idx_a], best_S[idx_b]))
                    if e in avail_C:
                        avail_C.discard(e)
                        
        # Cover remaining avail_C with pure cliques
        pure_cliques = cover_C_edges(C, avail_C)
        
        result = mixed_cliques + pure_cliques
        return result
    
    def cover_C_edges(C, edge_set):
        remaining = set(edge_set)
        cliques = []
        C_list = list(C)
        while remaining:
            best_clique = None
            best_count = 0
            for start in C_list:
                clique = [start]
                clique_set = {start}
                candidates = [v for v in C_list if v != start and frozenset((v, start)) in remaining]
                while candidates:
                    # Pick first candidate (simple greedy)
                    v = candidates.pop(0)
                    clique.append(v)
                    clique_set.add(v)
                    candidates = [w for w in candidates if frozenset((w, v)) in remaining]
                
                count = 0
                for e in remaining:
                    if e.issubset(clique_set):
                        count += 1
                if count > best_count:
                    best_count = count
                    best_clique = list(clique)
            
            if best_count == 0 or best_clique is None:
                break
            cliques.append(best_clique)
            for e in list(remaining):
                if e.issubset(set(best_clique)):
                    remaining.discard(e)
        return cliques
    
    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(solve(*spec))
    return results
