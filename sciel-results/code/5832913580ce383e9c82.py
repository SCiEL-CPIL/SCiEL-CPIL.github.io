"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        # Generate edges as per the contract
        # Edges within the clique K (first k vertices)
        edges_k = [(u, v) for u in range(k) for v in range(u + 1, k)]
        # Edges between K and I (rest of vertices)
        edges_cross = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges_cross.append((u, v))
        return edges_k, edges_cross

    def solve_instance(n, k, p, seed):
        # Reconstruct the graph structure
        # K = {0, ..., k-1}, I = {k, ..., n-1}
        # We need N(v) for each v in I
        # The instance function returns edges, but we can regenerate neighbors more directly 
        # or parse the edges. Parsing is safer to match the exact random sequence if we 
        # don't re-implement the RNG logic identically. 
        # However, the contract says "keep the seed program's generator". 
        # The seed program's `instance` returns `sorted(edges)`.
        # I will use the exact same logic to get the edges, then build adjacency.
        
        rng = Random(seed)
        # Recreate the edge set exactly as in the seed program's instance function
        edges = []
        # Edges in K
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        # Cross edges
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        
        # Build neighbor lists for I vertices
        # N[v] = set of neighbors in K
        N = [set() for _ in range(n)]
        # Track edges in K that are "covered" by the cliques formed from I vertices
        # An edge (u, v) in K is covered if there exists some w in I such that u in N[w] and v in N[w]
        
        # First, determine N[w] for all w in I
        for u, v in edges:
            if u < k and v >= k:
                N[v].add(u)
            elif v < k and u >= k:
                N[u].add(v)
            # Edges in K are u<k, v<k. They don't affect N.
            
        cliques = []
        
        # Step 1: Add cliques for each vertex in I
        # For each v in I, the clique is {v} U N[v]
        # This covers all edges (v, u) for u in N[v].
        # It also covers all edges (u1, u2) where u1, u2 in N[v].
        
        # We need to track which edges in K are already covered.
        # Let's use a set of edges in K.
        edges_in_K = set()
        for u in range(k):
            for v in range(u + 1, k):
                edges_in_K.add((u, v))
        
        covered_edges_in_K = set()
        
        for v in range(k, n):
            if len(N[v]) >= 2:
                clique = [v] + sorted(list(N[v]))
                cliques.append(clique)
                # Mark all edges within N[v] as covered
                nv = list(N[v])
                for i in range(len(nv)):
                    for j in range(i + 1, len(nv)):
                        u1, u2 = nv[i], nv[j]
                        if u1 > u2:
                            u1, u2 = u2, u1
                        covered_edges_in_K.add((u1, u2))
            elif len(N[v]) == 1:
                u = list(N[v])[0]
                cliques.append([u, v])
                # No edges in K covered by a clique of size 2 (one from I, one from K)
            # If len(N[v]) == 0, no edges to cover for this v, so no clique added.
        
        # Step 2: Cover remaining edges in K
        remaining_edges = edges_in_K - covered_edges_in_K
        
        # We need to partition remaining_edges into cliques.
        # Since the graph induced by remaining_edges on K is a subgraph of a complete graph,
        # we can find maximal cliques greedily.
        # Actually, a simpler and very effective strategy for the remainder of a complete graph:
        # The remaining edges form a graph H. We can color the vertices of H? No.
        # We want to cover edges of H with complete graphs.
        # Greedy: Find a maximal clique in H, add it, remove its edges.
        
        # To find maximal cliques efficiently in a small graph (k <= 90), we can use Bron-Kerbosch 
        # or just a simple greedy heuristic which is likely sufficient given the structure.
        # However, since H is a subgraph of K_k, a greedy approach of picking the vertex with 
        # highest degree in the remaining graph and adding its neighbors that are still connected 
        # might work.
        
        # Let's use a simple greedy clique cover for the remaining edges.
        # While remaining_edges is not empty:
        #   Pick an edge (u, v) from remaining_edges.
        #   Try to extend {u, v} to a maximal clique in the remaining graph.
        
        rem_adj = {u: set() for u in range(k)}
        for u, v in remaining_edges:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
            
        while remaining_edges:
            # Pick an arbitrary edge to start a clique
            # To be deterministic, pick the lexicographically smallest edge
            edge = min(remaining_edges)
            u, v = edge
            clique_vertices = {u, v}
            
            # Greedily add vertices to the clique
            # A vertex w can be added if it is adjacent to all current clique vertices in rem_adj
            changed = True
            while changed:
                changed = False
                # Candidates are neighbors of u that are not in clique
                candidates = list(rem_adj[u])
                # Sort for determinism
                candidates.sort()
                for w in candidates:
                    if w in clique_vertices:
                        continue
                    # Check if w is connected to all other vertices in clique_vertices
                    is_connected = True
                    for x in clique_vertices:
                        if x == w:
                            continue
                        if w not in rem_adj[x]:
                            is_connected = False
                            break
                    if is_connected:
                        clique_vertices.add(w)
                        changed = True
                        # Update rem_adj? No, we are just forming the clique. 
                        # We will remove all edges within clique_vertices at the end.
            
            # Convert clique to list
            clique_list = sorted(clique_vertices)
            if len(clique_list) >= 2:
                cliques.append(clique_list)
            
            # Remove all edges within clique_vertices from remaining_edges and rem_adj
            # Edges to remove: all pairs in clique_vertices
            edges_to_remove = set()
            cv_list = list(clique_vertices)
            for i in range(len(cv_list)):
                for j in range(i + 1, len(cv_list)):
                    a, b = cv_list[i], cv_list[j]
                    if a > b:
                        a, b = b, a
                    if (a, b) in remaining_edges:
                        edges_to_remove.add((a, b))
            
            for e in edges_to_remove:
                remaining_edges.remove(e)
                a, b = e
                if b in rem_adj[a]:
                    rem_adj[a].remove(b)
                if a in rem_adj[b]:
                    rem_adj[b].remove(a)

        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        n, k, p, seed = spec
        cliques = solve_instance(n, k, p, seed)
        results.append(cliques)
        
    return results
# EVOLVE-BLOCK-END
