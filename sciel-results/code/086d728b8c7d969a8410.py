def construct():
    from random import Random
    import time
    import sys
    from itertools import combinations

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = set()
        # Clique on first k vertices
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        # Cross edges
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def max_clique_in_graph(adj, vertices):
        """Find a maximal clique in the subgraph induced by vertices using greedy approach."""
        if not vertices:
            return []
        # Sort by degree in the subgraph
        deg = {v: len(adj[v] & vertices) for v in vertices}
        ordered = sorted(vertices, key=lambda x: -deg[x])
        clique = [ordered[0]]
        candidates = set(ordered[1:])
        for v in ordered[1:]:
            if v in candidates and all(c in adj[v] for c in clique):
                clique.append(v)
        return clique

    def solve(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Separate cross edges and C-C edges
        cross_neighbors = {}  # v (in I) -> set of C-neighbors
        cc_edges = set()  # C-C edges
        
        for (a, b) in edges:
            if a < k and b < k:
                cc_edges.add((min(a,b), max(a,b)))
            elif a >= k and b < k:
                if a not in cross_neighbors:
                    cross_neighbors[a] = set()
                cross_neighbors[a].add(b)
            elif a < k and b >= k:
                if b not in cross_neighbors:
                    cross_neighbors[b] = set()
                cross_neighbors[b].add(a)
        
        # Build CC adjacency for quick lookup
        cc_adj = {i: set() for i in range(k)}
        for (a, b) in cc_edges:
            cc_adj[a].add(b)
            cc_adj[b].add(a)
        
        # Track which C-C edges are covered
        covered_cc = set()
        cliques = []
        
        # Generate all possible mixed cliques: for each I-vertex v,
        # find all maximal cliques in the CC subgraph induced by cross_neighbors[v]
        # that only use uncovered CC edges.
        
        # We'll use a conflict-graph greedy approach:
        # Each candidate clique covers a set of CC edges.
        # Two candidates conflict if they share any CC edge.
        # We want to select a set of non-conflicting candidates that covers all cross edges.
        
        # Step 1: For each I-vertex, generate candidate C-subsets (cliques in CC graph using only uncovered edges)
        # To keep it tractable, we generate maximal cliques in the "available" CC subgraph.
        
        # Process I-vertices in order of decreasing neighborhood size
        i_vertices = sorted(cross_neighbors.keys(), key=lambda v: -len(cross_neighbors[v]))
        
        for v in i_vertices:
            S = set(cross_neighbors[v])
            if not S:
                continue
            
            # Find all maximal cliques in the CC subgraph induced by S, 
            # using only edges that are still uncovered
            # We'll do this iteratively: find a maximal clique, mark its edges as covered, repeat
            
            while S:
                # Build available subgraph on S: edge (a,b) exists if (a,b) in cc_edges and not in covered_cc
                avail_vertices = S
                avail_edges = set()
                for a in S:
                    for b in S:
                        if a < b and (a, b) in cc_edges and (a, b) not in covered_cc:
                            avail_edges.add((a, b))
                
                if not avail_edges:
                    # No available edges; we can only form 2-cliques (v, c) for each c
                    # Pick the c that maximizes future benefit (fewest remaining options)
                    # For now, just pick one
                    c = sorted(S)[0]
                    clique = [v, c]
                    cliques.append(clique)
                    S.discard(c)
                    continue
                
                # Find a maximal clique in the available subgraph
                # Use greedy: start with highest-degree vertex
                deg = {v_: 0 for v_ in S}
                for (a, b) in avail_edges:
                    deg[a] += 1
                    deg[b] += 1
                
                if not deg:
                    c = sorted(S)[0]
                    clique = [v, c]
                    cliques.append(clique)
                    S.discard(c)
                    continue
                    
                # Sort by degree, try to build clique
                ordered = sorted(S, key=lambda x: -deg[x])
                clique_c = [ordered[0]]
                for c in ordered[1:]:
                    # Check if c is connected to all in clique_c via avail_edges
                    ok = True
                    for c2 in clique_c:
                        pair = (min(c, c2), max(c, c2))
                        if pair not in avail_edges:
                            ok = False
                            break
                    if ok:
                        clique_c.append(c)
                
                if len(clique_c) < 2:
                    # Single vertex clique in C-part
                    c = ordered[0]
                    clique = [v, c]
                    cliques.append(clique)
                    S.discard(c)
                else:
                    clique = [v] + clique_c
                    cliques.append(clique)
                    # Mark CC edges within clique_c as covered
                    for i in range(len(clique_c)):
                        for j in range(i + 1, len(clique_c)):
                            pair = (min(clique_c[i], clique_c[j]), max(clique_c[i], clique_c[j]))
                            covered_cc.add(pair)
                    S -= set(clique_c)
        
        # Cover remaining CC edges with pure-C cliques
        remaining_cc = cc_edges - covered_cc
        if remaining_cc:
            rem_edges = set(remaining_cc)
            vertices_involved = set()
            for (a, b) in rem_edges:
                vertices_involved.add(a)
                vertices_involved.add(b)
            
            while rem_edges:
                # Find a maximal clique in the remaining graph
                deg = {v: 0 for v in vertices_involved}
                for (a, b) in rem_edges:
                    deg[a] = deg.get(a, 0) + 1
                    deg[b] = deg.get(b, 0) + 1
                
                if not deg:
                    break
                    
                start = max(deg, key=lambda x: deg[x])
                clique = [start]
                candidates = [v for v in vertices_involved if v != start]
                for v in sorted(candidates, key=lambda x: -deg[x]):
                    ok = True
                    for c in clique:
                        pair = (min(v, c), max(v, c))
                        if pair not in rem_edges:
                            ok = False
                            break
                    if ok:
                        clique.append(v)
                
                if len(clique) < 2:
                    a, b = next(iter(rem_edges))
                    clique = [a, b]
                
                cliques.append(clique)
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        pair = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                        rem_edges.discard(pair)
        
        # Convert to plain Python ints and lists
        result = []
        for clique in cliques:
            result.append([int(x) for x in clique])
        return result
    
    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        cliques = solve(*spec)
        results.append(cliques)
    
    return results
