def construct():
    from random import Random
    import itertools
    import time

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    for n, k, p, seed in specs:
        if time.time() > deadline:
            # Fallback: return empty or best effort if timeout
            # But we should not timeout normally.
            pass

        edges = instance(n, k, p, seed)
        
        # 1. Build adjacency for K-I
        neighbors_in_k = {v: set() for v in range(k, n)}
        # Also track K-K edges to know which are covered
        # We know all K-K edges exist.
        
        for u, v in edges:
            if u < k and v >= k:
                neighbors_in_k[v].add(u)
        
        cliques = []
        
        # 2. Add cliques for each v in I
        # Clique C_v = N(v) U {v}
        # This covers cross edges (u,v) and K-K edges (u1, u2) where u1, u2 in N(v)
        
        covered_kk_edges = set()
        
        for v in range(k, n):
            nbs = neighbors_in_k[v]
            if nbs:
                clique = list(nbs) + [v]
                cliques.append(clique)
                # Mark K-K edges within nbs as covered
                nbs_list = list(nbs)
                for i in range(len(nbs_list)):
                    for j in range(i+1, len(nbs_list)):
                        covered_kk_edges.add((nbs_list[i], nbs_list[j]))
        
        # 3. Identify remaining K-K edges
        remaining_edges = []
        for u in range(k):
            for w in range(u+1, k):
                if (u, w) not in covered_kk_edges:
                    remaining_edges.append((u, w))
        
        # 4. Cover remaining edges with cliques in K
        # We need to partition the set of remaining edges into cliques.
        # A clique in K is just a subset of vertices where all pairs are in remaining_edges?
        # No, a clique in the graph (K, remaining_edges) is a set of vertices S such that
        # for all u, v in S, (u,v) is in remaining_edges.
        # We want to minimize the number of such cliques that cover all remaining_edges.
        
        if remaining_edges:
            # Build graph of remaining edges on K
            # Vertices involved
            involved_vertices = set()
            for u, w in remaining_edges:
                involved_vertices.add(u)
                involved_vertices.add(w)
            
            # We need to find a clique cover of the graph G_rem = (involved_vertices, remaining_edges)
            # This is equivalent to finding a vertex coloring of the complement graph.
            # Given small k (10, 20, 30), we can use a simple greedy or exact backtracking.
            # Let's try greedy first: repeatedly find max clique, remove its edges, repeat.
            
            rem_edges_set = set(remaining_edges)
            
            # Function to check if a set of vertices is a clique in G_rem
            def is_clique(verts):
                verts_list = list(verts)
                for i in range(len(verts_list)):
                    for j in range(i+1, len(verts_list)):
                        u, w = verts_list[i], verts_list[j]
                        if u > w: u, w = w, u
                        if (u, w) not in rem_edges_set:
                            return False
                return True

            # Greedy Clique Cover
            # Sort vertices by degree in G_rem descending?
            # Or just iterate.
            
            # To be safe and optimal for small N, we can try to find max cliques.
            # But greedy is usually good enough for these sizes if we are careful.
            # Let's use a simple backtracking if the number of remaining edges is small,
            # otherwise greedy.
            
            # For k=30, max edges ~435. Remaining edges could be many.
            # Greedy:
            # 1. Find a maximal clique in G_rem.
            # 2. Add it to cliques.
            # 3. Remove all edges in this clique from G_rem.
            # 4. Repeat.
            
            # To find a maximal clique, we can use Bron-Kerbosch or just greedy expansion.
            # Greedy expansion:
            # Start with a vertex, add neighbors that form a clique with current set.
            
            # Let's implement a simple greedy max-clique finder.
            def find_max_clique(G, vertices):
                # G is a set of edges
                # vertices is a set of available vertices
                if not vertices:
                    return []
                
                # Heuristic: pick vertex with highest degree
                # Degrees within 'vertices'
                deg = {v: 0 for v in vertices}
                for u, w in G:
                    if u in vertices and w in vertices:
                        deg[u] += 1
                        deg[w] += 1
                
                # Pick v with max degree
                v_max = max(vertices, key=lambda x: deg[x])
                
                # Try to build clique starting with v_max
                # Candidates must be neighbors of v_max
                candidates = set()
                for u, w in G:
                    if u == v_max:
                        if w in vertices:
                            candidates.add(w)
                    elif w == v_max:
                        if u in vertices:
                            candidates.add(u)
                
                # Now find max clique in subgraph induced by candidates
                # This is recursive. For small sizes, it's fine.
                # To avoid deep recursion, we can just do a greedy pass.
                
                # Greedy pass on candidates
                current_clique = {v_max}
                available = candidates
                changed = True
                while changed:
                    changed = False
                    for v in list(available):
                        # Check if v can be added to current_clique
                        # v must be connected to all in current_clique
                        can_add = True
                        for u in current_clique:
                            if u == v: continue
                            edge = (min(u, v), max(u, v))
                            if edge not in G:
                                can_add = False
                                break
                        if can_add:
                            current_clique.add(v)
                            available.remove(v)
                            changed = True
                            
                # This greedy might not find the MAX clique, but a maximal one.
                # For minimization of number of cliques, Max Clique is better.
                # Given the constraints, a simple greedy is likely sufficient.
                return list(current_clique)

            # Actually, for "Minimize total number of cliques", we want LARGE cliques.
            # The greedy above finds a maximal clique.
            
            while rem_edges_set:
                if time.time() > deadline:
                    break
                # Find a maximal clique in the graph defined by rem_edges_set
                # Vertices involved in rem_edges_set
                verts = set()
                for u, w in rem_edges_set:
                    verts.add(u)
                    verts.add(w)
                
                # Use a simple greedy to find a large clique
                # Sort vertices by degree in rem_edges_set
                deg = {v: 0 for v in verts}
                for u, w in rem_edges_set:
                    deg[u] += 1
                    deg[w] += 1
                
                # Try starting with each high-degree vertex? 
                # Just take the first one for simplicity, or the best one.
                # Let's try to find a maximal clique using a simple algorithm.
                
                best_clique = []
                
                # We can just use the greedy expansion from the highest degree vertex
                v_start = max(verts, key=lambda x: deg[x])
                
                # Candidates: neighbors of v_start
                cands = set()
                for u, w in rem_edges_set:
                    if u == v_start:
                        cands.add(w)
                    elif w == v_start:
                        cands.add(u)
                
                clique = {v_start}
                avail = cands
                # Iteratively add vertices that connect to all in clique
                while True:
                    added = False
                    # Sort by degree in avail?
                    for v in list(avail):
                        ok = True
                        for u in clique:
                            if u == v: continue
                            edge = (min(u, v), max(u, v))
                            if edge not in rem_edges_set:
                                ok = False
                                break
                        if ok:
                            clique.add(v)
                            avail.remove(v)
                            added = True
                            # Restart loop to check new connections? 
                            # Actually, adding v doesn't make other non-neighbors into neighbors.
                            # It only removes v from avail.
                            # So we can just continue.
                    if not added:
                        break
                
                if not clique:
                    # Should not happen if rem_edges_set is not empty
                    break
                    
                cliques.append(sorted(clique))
                
                # Remove edges in this clique from rem_edges_set
                clique_list = list(clique)
                for i in range(len(clique_list)):
                    for j in range(i+1, len(clique_list)):
                        u, w = clique_list[i], clique_list[j]
                        edge = (min(u, w), max(u, w))
                        if edge in rem_edges_set:
                            rem_edges_set.remove(edge)
        
        # Ensure types are correct
        # Vertices are ints. Lists are lists.
        results.append(cliques)
        
    return results
