"""Strategy: Merge universal neighbors into the core clique to reduce clique count."""

def construct():
    from random import Random
    import time

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _solve_safe(n, k, p, seed, deadline):
        rng = Random(seed)
        adj = [set() for _ in range(n)]
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    adj[v].add(u)
        
        # available contains core edges (u, v) with u < v, u, v < k, not yet covered
        available = set((u, v) for u in range(k) for v in range(u + 1, k))
        covered_cross = set()
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
                
                cliques.append([i] + list(S))
                
                # Mark cross edges as covered
                for c in S:
                    covered_cross.add((c, i))
                
                # Remove core edges from available
                for u in S:
                    for v in S:
                        if u < v:
                            available.discard((u, v))
                
                N_i.difference_update(S)
            
            # If broke out of while due to timeout, cover remaining N_i with 2-cliques
            if N_i:
                for c in N_i:
                    if (c, i) not in covered_cross:
                        cliques.append([c, i])
                        covered_cross.add((c, i))
        
        # Now cover remaining core edges
        # available contains edges not covered by P_I.
        # We need to partition available into cliques.
        # Greedy clique partition
        remaining_edges = set(available)
        while remaining_edges:
            if time.time() > deadline:
                # Fallback: use 2-cliques for all remaining
                for e in remaining_edges:
                    cliques.append([e[0], e[1]])
                break
            
            # Find a maximal clique in the graph defined by remaining_edges
            # Vertices are 0..k-1
            # Build adjacency from remaining_edges
            adj_rem = [set() for _ in range(k)]
            for u, v in remaining_edges:
                adj_rem[u].add(v)
                adj_rem[v].add(u)
            
            # Greedy: pick vertex with highest degree, add neighbors that form clique
            # Simpler: pick first vertex, add all neighbors, then filter
            # Even simpler: just take any edge, then try to extend?
            # Since k is small, we can just find a clique.
            # Start with empty set, add vertices greedily.
            S = set()
            for v in range(k):
                can_add = True
                for u in S:
                    if v not in adj_rem[u]:
                        can_add = False
                        break
                if can_add:
                    S.add(v)
            
            if not S:
                # Should not happen if remaining_edges not empty
                break
            
            if len(S) < 2:
                # Single vertex covers no edges. Pick an edge.
                u, v = next(iter(remaining_edges))
                S = {u, v}
            
            cliques.append(list(S))
            
            # Remove edges in S from remaining_edges
            for u in S:
                for v in S:
                    if u < v:
                        remaining_edges.discard((u, v))
        
        return cliques

    results = []
    start_time = time.time()
    deadline = start_time + 100  # Safety margin

    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(_solve_safe(*spec, deadline))

    return results
