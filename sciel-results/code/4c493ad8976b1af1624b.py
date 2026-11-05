<<<<<<< SEARCH
    def solve_instance(n, p, seed):
        """Solve one instance: find minimum transversal."""
        edges = generate_graph(n, p, seed)
        adj = build_adjacency(n, edges)
        
        # Get all maximal cliques of size >= 2
        cliques = bron_kerbosch_maximal_cliques(adj, n)
        cliques = [c for c in cliques if len(c) >= 2]
        
        if not cliques:
            return []
        
        # Greedy set cover
        selected = greedy_transversal(cliques, n)
        
        # Local search improvement
        remaining_time = deadline - time.time()
        if remaining_time > 5:
            selected = local_search_improve(selected, cliques, n, max_iters=10000)
        
        return sorted(selected)
=======
    def solve_instance(n, p, seed):
        """Solve one instance: find minimum transversal using vertex cover on clique intersection graph."""
        edges = generate_graph(n, p, seed)
        adj = build_adjacency(n, edges)
        
        # Get all maximal cliques of size >= 2
        cliques = bron_kerbosch_maximal_cliques(adj, n)
        cliques = [c for c in cliques if len(c) >= 2]
        
        if not cliques:
            return []
        
        # Build clique intersection graph: nodes are cliques, edges connect cliques sharing vertices
        num_cliques = len(cliques)
        clique_adj = [set() for _ in range(num_cliques)]
        
        # For each vertex, find all cliques containing it
        vertex_to_cliques = defaultdict(list)
        for i, clique in enumerate(cliques):
            for v in clique:
                vertex_to_cliques[v].append(i)
        
        # Connect cliques that share a vertex
        for v, clique_list in vertex_to_cliques.items():
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    clique_adj[clique_list[i]].add(clique_list[j])
                    clique_adj[clique_list[j]].add(clique_list[i])
        
        # Find minimum vertex cover in clique intersection graph using greedy + local search
        # Start with greedy: pick clique with highest degree
        remaining = set(range(num_cliques))
        covered_cliques = set()
        selected_cliques = []
        
        while remaining:
            # Pick clique with max degree in remaining
            best_c = max(remaining, key=lambda c: len(clique_adj[c] & remaining))
            selected_cliques.append(best_c)
            covered_cliques.add(best_c)
            remaining -= clique_adj[best_c]
            remaining.discard(best_c)
        
        # Convert selected cliques to a hitting set: for each selected clique, pick a vertex
        # that covers the most other cliques
        selected_vertices = set()
        for c in selected_cliques:
            # Find vertex in this clique that covers most uncovered cliques
            best_v = None
            best_count = -1
            for v in cliques[c]:
                if v in selected_vertices:
                    continue
                # Count how many cliques in selected_cliques contain v
                count = sum(1 for sc in selected_cliques if v in cliques[sc])
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v is not None:
                selected_vertices.add(best_v)
        
        # Verify and improve with local search
        selected = list(selected_vertices)
        remaining_time = deadline - time.time()
        if remaining_time > 5:
            selected = local_search_improve(selected, cliques, n, max_iters=10000)
        
        return sorted(selected)
>>>>>>> REPLACE
