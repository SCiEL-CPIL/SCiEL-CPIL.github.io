import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, p, seed):
        import random
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []

    for i, spec in enumerate(specs):
        time_left = deadline - time.time()
        if time_left <= 0:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue

        edges = instance(*spec)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Map edge to index
        edge_list = list(edges)
        edge_index = {e: i for i, e in enumerate(edge_list)}
        m = len(edge_list)
        
        # 1. Calculate maximum matching for each vertex
        # For each vertex v, we have a set of incident edges.
        # We want to pair them up.
        # Since it's a star, max matching is floor(deg/2).
        # We just need to pick which pairs.
        # Any maximum matching works for the count, but the structure of the resulting
        # components (cycles vs paths) depends on the specific matching.
        # To minimize cycles, we might need to be careful?
        # Actually, the number of cycles N_c depends on the matching.
        # Can we choose matchings to minimize N_c?
        # This is a more complex problem.
        # However, for random graphs, the number of cycles is likely small.
        # Let's just pick arbitrary maximum matchings.
        
        # Store the partner for each edge at each endpoint
        # partner[edge_idx][endpoint] = partner_edge_idx or None
        partner = [{} for _ in range(m)]
        
        for v in range(n):
            incident_edges = [edge_index[e] for e in edge_list if v in e]
            # Shuffle to randomize
            import random
            random.shuffle(incident_edges)
            # Pair them up
            for j in range(0, len(incident_edges) - 1, 2):
                e1 = incident_edges[j]
                e2 = incident_edges[j+1]
                u1, v1 = edge_list[e1]
                u2, v2 = edge_list[e2]
                
                # Determine which endpoint is v
                if u1 == v:
                    p1 = v1
                else:
                    p1 = u1
                if u2 == v:
                    p2 = v2
                else:
                    p2 = u2
                    
                partner[e1][v] = e2
                partner[e2][v] = e1
                
        # 2. Build the edge graph
        # Nodes are edge indices 0..m-1
        # Edges connect e1 and e2 if they are paired at some vertex
        edge_graph = defaultdict(list)
        for e1 in range(m):
            for v in edge_list[e1]:
                if v in partner[e1]:
                    e2 = partner[e1][v]
                    edge_graph[e1].append(e2)
                    # Avoid duplicates if paired at both ends?
                    # If e1 and e2 are paired at v1 and v2, they are the same edge pair?
                    # No, e1 and e2 are distinct edges.
                    # They can be paired at one or both endpoints.
                    # If paired at both, they form a 2-cycle?
                    # A 2-cycle in the edge graph means e1-e2-e1.
                    # This corresponds to a path of length 2 in G?
                    # e1=(u,v), e2=(v,u)? No, simple graph.
                    # e1=(u,v), e2=(v,w).
                    # If paired at v, they are connected.
                    # If they are also paired at u? e2 must be incident to u.
                    # So e2=(u,w)? Then e1=(u,v), e2=(u,w).
                    # They share u.
                    # So they can be paired at u and v?
                    # If e1=(u,v) and e2=(u,v) (multigraph)? No.
                    # So they can only be paired at one vertex unless they form a triangle?
                    # No, two distinct edges share at most 1 vertex in a simple graph.
                    # So they are paired at most once.
                    
        # 3. Find components
        visited = [False] * m
        components = []
        
        for i in range(m):
            if not visited[i]:
                # BFS
                queue = [i]
                visited[i] = True
                comp = []
                while queue:
                    node = queue.pop(0)
                    comp.append(node)
                    for neighbor in edge_graph[node]:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            queue.append(neighbor)
                components.append(comp)
                
        # 4. Process components
        paths = []
        for comp in components:
            if not comp:
                continue
            # Determine if it's a cycle or a path
            # Count degrees in the component
            deg = {e: 0 for e in comp}
            for e in comp:
                for neighbor in edge_graph[e]:
                    if neighbor in deg:
                        deg[e] += 1
                        
            is_cycle = all(d == 2 for d in deg.values())
            
            if is_cycle:
                # Break the cycle
                # Pick an edge to break
                # The cycle is a sequence of edges.
                # We need to reconstruct the path of vertices.
                # Start at comp[0]
                # We need to traverse the cycle.
                # Since it's a cycle, we can just walk.
                # But we need to break it.
                # Let's find the sequence of vertices.
                # This is tricky with just edge indices.
                # Let's reconstruct the path.
                
                # To break, we just take the cycle, remove one edge,
                # and output the remaining path and the isolated edge.
                
                # Reconstruct path for the whole cycle first
                # Start with comp[0]
                # Find the two neighbors. Pick one to start.
                # This is a standard cycle traversal.
                
                # However, we need to output vertex lists.
                # Let's find the vertex sequence.
                
                # Pick a start edge
                start_e = comp[0]
                u, v = edge_list[start_e]
                
                # We need to determine the direction.
                # Let's try to walk from u.
                # The next edge must be incident to u? No, start_e is incident to u and v.
                # The partner of start_e at u is the next edge?
                # If start_e is paired at u with e_next, then the path goes ...-e_next-u-start_e-...
                
                # Let's build the list of edges in order.
                # Since it's a cycle, we can just pick an arbitrary order.
                # But we need the vertex order.
                
                # Let's find the sequence of edges.
                # Start at start_e.
                # Neighbors are partner[start_e][u] and partner[start_e][v].
                # Let's pick partner[start_e][u] as the next edge.
                # This implies the path enters start_e from u.
                # So the previous edge is partner[start_e][u].
                # And start_e goes to v.
                
                # So the sequence of vertices is ... - u - v - ...
                
                # Let's trace the full cycle.
                # We need to be careful with the direction.
                
                # Let's just extract the path by walking.
                # Start at vertex u.
                # Current edge is start_e.
                # We came from... nowhere?
                # Let's say we start at u, take start_e to v.
                # At v, we take the partner of start_e at v.
                
                path_vertices = [u, v]
                current_e = start_e
                current_v = v
                prev_v = u
                
                # We need to continue until we return to u?
                # But we want to break it.
                # So we stop when we are about to take the edge that leads back to u?
                # Or just walk the whole cycle and then split.
                
                # Let's walk the whole cycle.
                # We have a set of edges in the component.
                # We can just use a visited set for edges.
                
                visited_edges_in_cycle = set()
                visited_edges_in_cycle.add(current_e)
                
                while True:
                    # Find next edge
                    next_e = None
                    # The partner of current_e at current_v
                    if current_v in partner[current_e]:
                        next_e = partner[current_e][current_v]
                    else:
                        # Should not happen in a cycle
                        break
                        
                    if next_e in visited_edges_in_cycle:
                        # We closed the loop
                        break
                        
                    visited_edges_in_cycle.add(next_e)
                    u2, v2 = edge_list[next_e]
                    next_v = u2 if v2 == current_v else v2
                    path_vertices.append(next_v)
                    
                    prev_v = current_v
                    current_v = next_v
                    current_e = next_e
                    
                # path_vertices is the full cycle vertices.
                # e.g., [u, v, w, x, u]
                # We need to break it.
                # Remove the last edge (which connects last vertex to first).
                # The path is path_vertices[:-1]
                # The isolated edge is (path_vertices[-2], path_vertices[-1])?
                # No, the edge is between path_vertices[-1] and path_vertices[0].
                
                path1 = path_vertices[:-1]
                # The isolated edge is (path_vertices[-1], path_vertices[0])
                # But wait, path_vertices[0] is u, path_vertices[-1] is the vertex before u.
                # The edge is (path_vertices[-1], u).
                # Is this edge in the graph? Yes.
                # So path2 = [path_vertices[-1], u]
                
                # But we must ensure path1 is a valid path.
                # path1 uses edges corresponding to the first L-1 steps.
                # The last step was the one we broke.
                
                # So we output path1 and path2.
                paths.append(path1)
                paths.append([path_vertices[-1], path_vertices[0]])
                
            else:
                # It's a path
                # Find the endpoints (degree 1)
                endpoints = [e for e in comp if deg[e] == 1]
                if not endpoints:
                    # Single edge?
                    e = comp[0]
                    u, v = edge_list[e]
                    paths.append([u, v])
                else:
                    # Start at one endpoint
                    start_e = endpoints[0]
                    u, v = edge_list[start_e]
                    # Determine which one is the end (degree 1)
                    # Both are degree 1.
                    # Let's pick u.
                    # We need to check if u has a partner.
                    # If not, u is the end.
                    if u not in partner[start_e]:
                        start_v = u
                        end_v = v
                    else:
                        start_v = v
                        end_v = u
                        
                    path_vertices = [start_v, end_v]
                    current_e = start_e
                    current_v = end_v
                    prev_v = start_v
                    
                    visited_edges_in_path = set()
                    visited_edges_in_path.add(current_e)
                    
                    while True:
                        next_e = None
                        if current_v in partner[current_e]:
                            next_e = partner[current_e][current_v]
                        else:
                            break
                            
                        if next_e in visited_edges_in_path:
                            break
                            
                        visited_edges_in_path.add(next_e)
                        u2, v2 = edge_list[next_e]
                        next_v = u2 if v2 == current_v else v2
                        path_vertices.append(next_v)
                        
                        prev_v = current_v
                        current_v = next_v
                        current_e = next_e
                        
                    paths.append(path_vertices)
                    
        results.append(paths)

    return results
