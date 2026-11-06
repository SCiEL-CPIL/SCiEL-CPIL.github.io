def construct():
    import random
    import sys
    sys.setrecursionlimit(10000)

    def instance_edges(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_min_vertex_cover(n, edges):
        # Build adjacency list
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # If no edges, cover is empty
        if not edges:
            return []

        best_cover = list(range(n)) # Worst case: all vertices
        
        # Helper to get a lower bound on additional vertices needed
        # A simple lower bound is the size of a maximum matching in the remaining graph.
        # For random sparse graphs, a greedy matching is a decent lower bound.
        def greedy_matching_lower_bound(adj_rem, verts_rem):
            # adj_rem is a dict or list of sets for the subgraph
            # verts_rem is the set of active vertices
            # We need to be careful with recursion depth and speed.
            # A simpler LB: ceil(number_of_edges / max_degree) is weak.
            # Let's use a greedy maximal matching.
            matched = [False] * n
            count = 0
            # Iterate through edges? We don't have the edge list for subgraph easily.
            # We can iterate through vertices and find an unmatched neighbor.
            for u in verts_rem:
                if matched[u]:
                    continue
                for v in adj_rem[u]:
                    if v in verts_rem and not matched[v]:
                        count += 1
                        matched[u] = True
                        matched[v] = True
                        break
            return count

        def branch_and_bound(cover, verts_rem, adj_rem, current_size):
            nonlocal best_cover
            
            # Pruning: if current_size >= len(best_cover), stop
            if current_size >= len(best_cover):
                return

            # Check if any edges remain
            # An edge remains if there is any vertex in verts_rem with a neighbor in verts_rem
            has_edges = False
            for u in verts_rem:
                if adj_rem[u] & verts_rem:
                    has_edges = True
                    break
            
            if not has_edges:
                # All edges covered
                if current_size < len(best_cover):
                    best_cover = cover[:]
                return

            # Lower bound
            lb = greedy_matching_lower_bound(adj_rem, verts_rem)
            if current_size + lb >= len(best_cover):
                return

            # Heuristic: Pick a vertex with high degree in the remaining graph
            # to branch on.
            max_deg = -1
            pivot = -1
            for u in verts_rem:
                deg = len(adj_rem[u] & verts_rem)
                if deg > max_deg:
                    max_deg = deg
                    pivot = u
            
            if pivot == -1:
                # No edges? Should have been caught above.
                if current_size < len(best_cover):
                    best_cover = cover[:]
                return

            # Branch 1: Include pivot in cover
            # Remove pivot and all edges incident to it.
            new_cover = cover + [pivot]
            new_verts_rem = set(verts_rem)
            new_verts_rem.remove(pivot)
            
            # We need to update adj_rem for the subgraph.
            # To avoid copying large structures, we can pass a mask or just rely on the 'verts_rem' set
            # and the original adj list, checking membership in verts_rem.
            # However, adj_rem in the function signature implies a modified graph.
            # Let's stick to using the original `adj` and `verts_rem` to define the active graph.
            # The "edges" are implicitly defined by pairs (u,v) in verts_rem where v in adj[u].
            
            # Recurse with new_verts_rem
            # Note: We don't need to modify adj, just the set of active vertices.
            branch_and_bound(new_cover, new_verts_rem, adj, current_size + 1)

            # Branch 2: Exclude pivot from cover
            # This means ALL neighbors of pivot MUST be in the cover.
            # Add all neighbors of pivot (that are in verts_rem) to the cover.
            # Remove pivot and all its neighbors from verts_rem.
            
            neighbors_in_rem = adj[pivot] & verts_rem
            if not neighbors_in_rem:
                # If no neighbors, excluding pivot doesn't force anything, 
                # but pivot must be removed? No, if we exclude pivot, we just say it's not in cover.
                # But if it has no edges in the remaining graph, it doesn't affect coverage.
                # We can just remove it from consideration.
                new_cover = cover[:]
                new_verts_rem = set(verts_rem)
                new_verts_rem.remove(pivot)
                branch_and_bound(new_cover, new_verts_rem, adj, current_size)
            else:
                new_cover = cover + list(neighbors_in_rem)
                new_verts_rem = set(verts_rem)
                new_verts_rem.remove(pivot)
                for v in neighbors_in_rem:
                    new_verts_rem.remove(v)
                
                branch_and_bound(new_cover, new_verts_rem, adj, current_size + len(neighbors_in_rem))

        branch_and_bound([], set(range(n)), adj, 0)
        
        return sorted(best_cover)

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        edges = instance_edges(n, p, seed)
        sol = solve_min_vertex_cover(n, edges)
        out.append(sol)
        
    return out
