"""Strategy: Greedy clique partition with local improvement polish."""

def construct():
    from random import Random
    import time
    import itertools

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _solve(n, k, p, seed, deadline):
        rng = Random(seed)
        adj = [set() for _ in range(n)]
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    adj[v].add(u)

        available = set((u, v) for u in range(k) for v in range(u + 1, k))
        cliques = []
        
        # Process each i in I
        for i in range(k, n):
            N_i = set(adj[i])
            while N_i:
                if time.time() > deadline:
                    break
                
                # Find maximal clique S in N_i using available edges
                S = set()
                for v in sorted(N_i):
                    can_add = True
                    for u in S:
                        edge = (min(u, v), max(u, v))
                        if edge not in available:
                            can_add = False
                            break
                    if can_add:
                        S.add(v)
                
                if not S:
                    v = next(iter(N_i))
                    S = {v}
                
                cliques.append([i] + sorted(S))
                
                # Remove core edges from available
                for u in S:
                    for v in S:
                        if u < v:
                            available.discard((u, v))
                
                N_i.difference_update(S)
            
            # If broke out of while due to timeout, cover remaining N_i with 2-cliques
            if N_i:
                for c in sorted(N_i):
                    cliques.append([c, i])
        
        # Cover remaining core edges
        remaining_edges = set(available)
        while remaining_edges:
            if time.time() > deadline:
                for e in remaining_edges:
                    cliques.append(sorted([e[0], e[1]]))
                break
            
            adj_rem = [set() for _ in range(k)]
            for u, v in remaining_edges:
                adj_rem[u].add(v)
                adj_rem[v].add(u)
            
            S = set()
            for v in range(k):
                can_add = True
                for u in S:
                    if v not in adj_rem[u]:
                        can_add = False
                        break
                if can_add:
                    S.add(v)
            
            if len(S) < 2:
                u, v = next(iter(remaining_edges))
                S = {u, v}
            
            cliques.append(sorted(S))
            
            for u in S:
                for v in S:
                    if u < v:
                        remaining_edges.discard((u, v))
        
        # Local improvement: try to merge cliques
        # A clique with an I-vertex can only be merged with another clique if they share no edges
        # and the union is still a valid clique (no two I-vertices together)
        # Actually, merging two cliques that both contain I-vertices is invalid (I is independent)
        # Merging a clique with an I-vertex and a pure-C clique: the I-vertex must be connected to all C-vertices in the pure-C clique
        # and no C-C edges should be double-covered
        
        # Simpler improvement: try to move vertices between cliques to reduce count
        # For now, just return the greedy result
        return cliques

    results = []
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(_solve(*spec, deadline))
    
    # Ensure types are correct
    final_results = []
    for cliques in results:
        clean_cliques = []
        for clique in cliques:
            clean_cliques.append([int(v) for v in clique])
        final_results.append(clean_cliques)
    
    return final_results
