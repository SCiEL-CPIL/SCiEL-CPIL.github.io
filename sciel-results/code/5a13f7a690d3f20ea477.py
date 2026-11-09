"""Program: minimize total number of cliques in an edge-exact clique partition of two dense random graphs."""
# EVOLVE-BLOCK-START
def construct():
    def random_graph(n, p, seed):
        import random
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def partition_cliques(edges):
        if not edges:
            return []
        
        n = max(max(e) for e in edges) + 1 if edges else 0
        adj_mask = [0] * n
        
        for u, v in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        
        uncovered = set(edges)
        cliques = []
        
        while uncovered:
            # Find the edge with the highest "clique density"
            # Heuristic: Max number of common neighbors in the uncovered subgraph
            best_edge = None
            max_common = -1
            
            # To speed this up, we can sample or just iterate.
            # With N=64, E~1000. Iterating all edges is fine.
            for e in uncovered:
                u, v = e
                # Common neighbors in the FULL graph
                common_full = adj_mask[u] & adj_mask[v]
                # We need common neighbors connected by UNCOVERED edges to u and v?
                # Actually, for the clique to be valid, all edges in the clique must be in `uncovered`.
                # So, w is a candidate if (u,w) in uncovered AND (v,w) in uncovered.
                # This is equivalent to: w is in adj_mask[u] AND adj_mask[v] 
                # AND (u,w) in uncovered AND (v,w) in uncovered.
                
                # Efficient check:
                # Candidates must be in common_full.
                # We can iterate through bits of common_full.
                count = 0
                mask = common_full
                while mask:
                    lsb = mask & -mask
                    w = lsb.bit_length() - 1
                    if (min(u, w), max(u, w)) in uncovered and (min(v, w), max(v, w)) in uncovered:
                        count += 1
                    mask ^= lsb
                
                if count > max_common:
                    max_common = count
                    best_edge = e
            
            if best_edge is None:
                # Fallback if logic fails
                e = next(iter(uncovered))
                best_edge = e
            
            u, v = best_edge
            
            # Build maximal clique containing u, v using only uncovered edges
            # Candidates are vertices w such that (u,w) and (v,w) are in uncovered
            # and w is connected to all future additions.
            
            clique = [u, v]
            
            # Initial candidates: neighbors of u and v in uncovered graph
            # Note: adj_mask is for full graph. We must filter by uncovered.
            
            # Helper to get neighbors of x in uncovered graph
            def get_uncovered_neighbors(x):
                neighbors = set()
                mask = adj_mask[x]
                while mask:
                    lsb = mask & -mask
                    y = lsb.bit_length() - 1
                    if (min(x, y), max(x, y)) in uncovered:
                        neighbors.add(y)
                    mask ^= lsb
                return neighbors
            
            cand_u = get_uncovered_neighbors(u)
            cand_v = get_uncovered_neighbors(v)
            
            current_candidates = cand_u.intersection(cand_v)
            
            # Greedy expansion
            while current_candidates:
                # Pick candidate w that maximizes connections to other candidates
                # to ensure we can keep growing the clique
                best_w = None
                max_conn = -1
                
                for w in current_candidates:
                    # Count how many other candidates are connected to w via uncovered edges
                    conn_count = 0
                    for other in current_candidates:
                        if other == w:
                            continue
                        if (min(w, other), max(w, other)) in uncovered:
                            conn_count += 1
                    if conn_count > max_conn:
                        max_conn = conn_count
                        best_w = w
                
                if best_w is None:
                    break
                    
                clique.append(best_w)
                
                # Update candidates:
                # New candidates must be connected to best_w via uncovered edges
                # and be in the previous candidate set
                new_candidates = set()
                for w in current_candidates:
                    if w == best_w:
                        continue
                    if (min(best_w, w), max(best_w, w)) in uncovered:
                        new_candidates.add(w)
                        
                current_candidates = new_candidates
            
            # Add clique and remove edges
            if len(clique) >= 2:
                cliques.append(sorted(clique))
                for i in range(len(clique)):
                    a = clique[i]
                    for j in range(i + 1, len(clique)):
                        b = clique[j]
                        e = (min(a, b), max(a, b))
                        uncovered.discard(e)
            else:
                # Safety: if no clique formed (shouldn't happen as we start with an edge)
                e = next(iter(uncovered))
                cliques.append([min(e), max(e)])
                uncovered.discard(e)
                
        return cliques

    result = []
    for (n, p, seed) in ((32, 0.6, 601), (64, 0.55, 602)):
        edges = random_graph(n, p, seed)
        cliques = partition_cliques(edges)
        result.append(cliques)
    return result
# EVOLVE-BLOCK-END
