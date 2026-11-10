def construct():
    from random import Random
    import itertools

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve(n, k, p, seed):
        edges = instance(n, k, p, seed)
        C = list(range(k))
        I = list(range(k, n))
        
        # cross[i] = list of c in C connected to i
        cross = {i: [] for i in I}
        for (a, b) in edges:
            if a < k and b < k:
                continue
            elif a >= k and b >= k:
                continue
            else:
                if a < k:
                    c, i = a, b
                else:
                    c, i = b, a
                cross[i].append(c)
        
        # Set of all cross edges, represented as (i, c)
        # We need to cover all these.
        uncovered_cross = set((i, c) for i in I for c in cross[i])
        used_C_edges = set()
        
        cliques = []
        
        # Helper: Maximal clique in a graph defined by allowed pairs
        # Graph nodes are vertices from C. Edge (u,v) exists if (u,v) not in used_C_edges
        # We want a clique S such that all pairs in S are allowed.
        # This is equivalent to finding a clique in the graph G_allowed.
        
        def find_max_clique(nodes, used_edges):
            # nodes: list of ints
            # used_edges: set of frozensets
            # Returns a list representing a maximal clique
            # Using Bron-Kerbosch or simple greedy?
            # For n<=30, a simple recursive branch and bound or even greedy might suffice.
            # Let's try a greedy approach first for speed, then refine if needed.
            # Actually, since we want to maximize coverage, we want the *largest* clique.
            
            # Build adjacency for the subgraph induced by 'nodes' where edge exists if NOT used
            # adj[v] = set of u in nodes such that {u,v} not in used_edges
            adj = {v: set() for v in nodes}
            for v in nodes:
                for u in nodes:
                    if u == v: continue
                    if frozenset((u, v)) not in used_edges:
                        adj[v].add(u)
                        
            # Greedy Max Clique:
            # Sort nodes by degree in this subgraph
            sorted_nodes = sorted(nodes, key=lambda x: -len(adj[x]))
            clique = []
            candidates = set(sorted_nodes)
            
            # Simple greedy: pick node with max degree in candidates, add to clique, remove non-neighbors
            while candidates:
                # Pick node with max degree in candidates
                v = max(candidates, key=lambda x: len(adj[x] & candidates))
                clique.append(v)
                # New candidates must be neighbors of v in the current candidate set
                candidates = candidates & adj[v]
                
            return clique

        while uncovered_cross:
            best_clique_size = 0
            best_i = None
            best_S = None
            
            for i in I:
                # Get remaining cross neighbors for i
                rem_neighbors = [c for (i2, c) in uncovered_cross if i2 == i]
                if not rem_neighbors:
                    continue
                
                # Find max clique in rem_neighbors with respect to used_C_edges
                S = find_max_clique(rem_neighbors, used_C_edges)
                
                if len(S) > best_clique_size:
                    best_clique_size = len(S)
                    best_i = i
                    best_S = S
            
            if best_i is None:
                # Should not happen if uncovered_cross is not empty
                break
                
            # Add clique {best_i} U best_S
            clique_list = [best_i] + best_S
            cliques.append(clique_list)
            
            # Update uncovered_cross
            for c in best_S:
                uncovered_cross.discard((best_i, c))
                
            # Update used_C_edges
            for u, v in itertools.combinations(best_S, 2):
                used_C_edges.add(frozenset((u, v)))
                
        # Now cover remaining C-edges
        # Total C-edges: all pairs in C
        all_C_edges = set(frozenset((u, v)) for u in range(k) for v in range(u+1, k))
        remaining_C_edges = all_C_edges - used_C_edges
        
        # Partition remaining_C_edges into cliques
        # Graph H on C with edges remaining_C_edges
        # Greedy: find max clique, remove its edges, repeat
        
        pure_cliques = []
        rem_edges = set(remaining_C_edges)
        
        # To find max clique in H, we can use the same find_max_clique logic
        # But the graph is changing (edges removed)
        
        while rem_edges:
            # Get nodes involved in rem_edges
            nodes = set()
            for e in rem_edges:
                nodes.update(e)
            nodes = list(nodes)
            
            # Find max clique in the graph induced by nodes with edges rem_edges
            # We need a version of find_max_clique that takes an explicit edge set
            def find_max_clique_explicit(nodes, edge_set):
                if not nodes: return []
                # Build adj
                adj = {v: set() for v in nodes}
                for u in nodes:
                    for v in nodes:
                        if u == v: continue
                        if frozenset((u, v)) in edge_set:
                            adj[u].add(v)
                            
                sorted_nodes = sorted(nodes, key=lambda x: -len(adj[x]))
                clique = []
                candidates = set(sorted_nodes)
                
                while candidates:
                    v = max(candidates, key=lambda x: len(adj[x] & candidates))
                    clique.append(v)
                    candidates = candidates & adj[v]
                return clique
            
            S = find_max_clique_explicit(nodes, rem_edges)
            
            if not S:
                # Should not happen
                break
                
            pure_cliques.append(S)
            
            # Remove edges within S from rem_edges
            for u, v in itertools.combinations(S, 2):
                rem_edges.discard(frozenset((u, v)))
                
        result = cliques + pure_cliques
        return result

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(solve(*spec))
    return results
