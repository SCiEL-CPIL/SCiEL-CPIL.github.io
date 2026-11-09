"""Program: minimize total number of cliques in an edge-exact clique partition of two dense random graphs."""
# EVOLVE-BLOCK-START
def construct():
    def random_graph(n, p, seed):
        import random
        rng = random.Random(seed)
        edges = set()
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        return edges, adj

    def partition_cliques(edges, adj, n):
        uncovered = set(edges)
        cliques = []
        
        while uncovered:
            # Find the edge with the maximum "clique potential"
            # Heuristic: Maximize number of common neighbors.
            # Tie-break: Maximize sum of degrees of common neighbors.
            
            best_edge = None
            max_common = -1
            best_common_set = None
            best_score = -1
            
            for e in uncovered:
                u, v = e
                # Common neighbors of u and v
                common = adj[u].intersection(adj[v])
                cnt = len(common)
                
                if cnt > max_common:
                    max_common = cnt
                    best_edge = e
                    best_common_set = common
                    # Calculate score for tie-breaking
                    best_score = sum(len(adj[x]) for x in common)
                elif cnt == max_common:
                    # Tie-break: pick the one with higher sum of degrees of common neighbors
                    score = sum(len(adj[x]) for x in common)
                    if score > best_score:
                        best_edge = e
                        best_common_set = common
                        best_score = score
            
            if best_edge is None:
                # Should not happen if uncovered is not empty
                break

            u, v = best_edge
            # Build a clique starting with {u, v}
            # Candidates are best_common_set
            candidates = set(best_common_set)
            clique = [u, v]
            
            while candidates:
                # Pick candidate with max degree within candidates
                best_c = None
                max_deg_in_cand = -1
                for c in candidates:
                    # Count neighbors in candidates
                    deg = 0
                    for d in candidates:
                        if c != d and d in adj[c]:
                            deg += 1
                    if deg > max_deg_in_cand:
                        max_deg_in_cand = deg
                        best_c = c
                
                if best_c is None:
                    break
                
                clique.append(best_c)
                
                # Update candidates: keep only those connected to best_c
                new_candidates = set()
                for c in candidates:
                    if c != best_c and best_c in adj[c]:
                        new_candidates.add(c)
                candidates = new_candidates

            # Emit clique
            if len(clique) >= 2:
                cliques.append(sorted(clique))
                # Remove all edges within clique from uncovered
                for i in range(len(clique)):
                    a = clique[i]
                    for j in range(i + 1, len(clique)):
                        b = clique[j]
                        e = (min(a, b), max(a, b))
                        uncovered.discard(e)
            else:
                # Fallback: single edge
                cliques.append(sorted([u, v]))
                uncovered.discard((u, v))

        return cliques

    result = []
    for (n, p, seed) in ((32, 0.6, 601), (64, 0.55, 602)):
        edges, adj = random_graph(n, p, seed)
        cliques = partition_cliques(edges, adj, n)
        result.append(cliques)
    return result
# EVOLVE-BLOCK-END
