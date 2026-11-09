def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def split_graph_partition(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set((min(u, v), max(u, v)) for u, v in edges)
        
        C = list(range(k))
        I = list(range(k, n))
        
        # Build adjacency for I vertices to C
        # neighbors_in_C[i] is a set of vertices in C connected to i
        neighbors_in_C = {i: set() for i in I}
        for u, v in edge_set:
            if u < k and v >= k:
                neighbors_in_C[v].add(u)
            elif v < k and u >= k:
                neighbors_in_C[u].add(v)
        
        # Generate all maximal cliques
        # 1. The clique C itself
        maximal_cliques = [C]
        
        # 2. For each i in I, N(i) U {i}
        for i in I:
            # N(i) is neighbors_in_C[i]
            # The clique is N(i) U {i}
            clique = list(neighbors_in_C[i]) + [i]
            if len(clique) >= 2: # Only add if it's a valid clique (size >= 2)
                maximal_cliques.append(clique)
        
        # Precompute edges for each maximal clique
        # clique_edges[j] is a set of edges in maximal_cliques[j]
        clique_edges = []
        for clique in maximal_cliques:
            es = set()
            for idx_a in range(len(clique)):
                for idx_b in range(idx_a + 1, len(clique)):
                    u = clique[idx_a]
                    v = clique[idx_b]
                    es.add((min(u, v), max(u, v)))
            clique_edges.append(es)
        
        remaining_edges = set(edge_set)
        partition = []
        
        # Greedy Set Cover / Partition
        # To optimize, we can track remaining edges count for each clique
        # But n is small, so simple loop is fine.
        
        # We need to pick the clique that covers the most remaining edges.
        # We iterate until no edges remain.
        
        # Optimization: If a clique's remaining edge count becomes 0, we can skip it.
        # But we must re-evaluate after each pick.
        
        while remaining_edges:
            best_clique_idx = -1
            best_count = 0
            
            for idx, es in enumerate(clique_edges):
                # Count intersection with remaining_edges
                # Since we remove edges, we can just count how many edges in es are in remaining_edges
                count = 0
                for e in es:
                    if e in remaining_edges:
                        count += 1
                        if count > best_count:
                            # Early exit if we can't beat best? 
                            # Max possible is len(es). 
                            # If count > best_count, we update.
                            pass 
                
                # The above loop is O(|E_clique|). 
                # We can do: len(es.intersection(remaining_edges))
                # But intersection creates a new set. 
                # For small sets, it's fine.
                
                count = len(es.intersection(remaining_edges))
                
                if count > best_count:
                    best_count = count
                    best_clique_idx = idx
            
            if best_count == 0:
                # This should not happen if the graph is covered by maximal cliques.
                # But just in case, pick any remaining edge as a clique of size 2?
                # No, the problem guarantees a solution exists.
                # If this happens, it means our maximal clique list was incomplete.
                # But for split graphs, C and N(i)+{i} cover all edges.
                break
            
            # Add the chosen clique to partition
            # We must return a list of ints.
            # The clique is already a list of ints.
            partition.append(maximal_cliques[best_clique_idx])
            
            # Remove covered edges
            remaining_edges -= clique_edges[best_clique_idx]
            
        return partition

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        cliques = split_graph_partition(n, k, p, seed)
        results.append(cliques)
        
    return results
