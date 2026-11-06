"""Path decomposition: Hopcroft-Karp maximum matching + local search + ILP polish."""
import time
import random
from collections import defaultdict, deque
import numpy as np


def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def get_adjacency(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def hopcroft_karp_matching(edges, adj):
        """
        Find maximum matching in the bipartite graph where:
        - Left side: directed edge (u,v) -> node index
        - Right side: directed edge (v,u) -> node index
        - Edge between left (u,v) and right (v,w) if v is common endpoint and u!=w
        
        Actually, let's use the standard formulation:
        Each undirected edge e=(u,v) creates two directed stubs: u->v and v->u.
        We want to match stubs so that if u->v is matched with v->w, then edges (u,v) and (v,w) are consecutive.
        
        Bipartite graph:
        - Left nodes: all directed stubs (u,v) for each undirected edge {u,v}
        - Right nodes: all directed stubs (v,u) for each undirected edge {u,v}
        
        Wait, that's not quite right. Let me think again.
        
        Standard path decomposition via matching:
        - Create a bipartite graph B = (L, R, E_B)
        - L = { (u,v) : {u,v} in E } (directed stubs, but we treat each undirected edge as giving two directed versions)
        - R = { (v,u) : {u,v} in E } 
        - Actually, simpler: L and R are both the set of "edge endpoints"
        
        Let me use a cleaner formulation:
        - For each undirected edge e_i = {u_i, v_i}, create two nodes: 
          left_i_u = (i, u) meaning "edge i, endpoint u" 
          left_i_v = (i, v) meaning "edge i, endpoint v"
        - Similarly right nodes.
        
        Actually, the cleanest way:
        - Left side: one node per directed edge. For undirected edge {u,v}, we have directed edges (u,v) and (v,u).
        - Right side: same set of directed edges.
        - There is an edge from left node (u,v) to right node (v,w) if v is the common vertex and u != w (to avoid using the same edge twice in a path, though since we're matching, each directed edge can only be used once).
        
        Wait, I think I'm overcomplicating. Let me use the standard approach:
        
        For path decomposition:
        - Each edge in the path decomposition uses two "half-edges" or "stubs"
        - A matching pairs stubs from different edges that meet at a common vertex
        - The number of paths = |E| - |M| where M is the matching
        
        Bipartite graph:
        - Left: all (edge_id, endpoint) pairs where endpoint is one of the two vertices of the edge
        - Right: all (edge_id, endpoint) pairs
        - Edge between left (e1, v) and right (e2, v) if e1 != e2 and v is an endpoint of both e1 and e2
        
        This is a general matching problem, not bipartite in the usual sense.
        
        Actually, it IS bipartite if we separate by edge:
        - Left: for each edge e, its two endpoints
        - Right: for each edge e, its two endpoints
        - But we need to ensure we don't match an edge's own endpoints to each other
        
        Let me just implement it directly:
        """
        # Build the bipartite graph for maximum matching
        # Left nodes: (edge_index, vertex) for each endpoint of each edge
        # Right nodes: (edge_index, vertex) for each endpoint of each edge
        # Edge between left (ei, v) and right (ej, v) if ei != ej and v is endpoint of both
        
        edge_list = list(edges)
        num_edges = len(edge_list)
        
        # Map each edge to its index
        edge_to_idx = {e: i for i, e in enumerate(edge_list)}
        
        # Build adjacency for the bipartite matching
        # Left side: (edge_idx, vertex)
        # Right side: (edge_idx, vertex)
        # We'll represent left nodes as (edge_idx * 2 + 0) for first endpoint, (edge_idx * 2 + 1) for second
        # Actually, let's just use a dict-based approach
        
        left_nodes = []
        right_nodes = []
        left_to_right = defaultdict(set)
        
        # For each edge, add its two endpoints as left and right nodes
        for ei, (u, v) in enumerate(edge_list):
            lu = ('L', ei, u)
            lv = ('L', ei, v)
            ru = ('R', ei, u)
            rv = ('R', ei, v)
            left_nodes.append(lu)
            left_nodes.append(lv)
            right_nodes.append(ru)
            right_nodes.append(rv)
        
        # Build edges in the bipartite graph
        # For each vertex v, find all edges incident to v
        vertex_to_edges = defaultdict(list)
        for ei, (u, v) in enumerate(edge_list):
            vertex_to_edges[u].append(ei)
            vertex_to_edges[v].append(ei)
        
        # For each vertex v, connect all left endpoints of edges incident to v with all right endpoints of edges incident to v (except same edge)
        for v, incident_edges in vertex_to_edges.items():
            for ei in incident_edges:
                # Left endpoint of edge ei at vertex v
                left_node = ('L', ei, v)
                for ej in incident_edges:
                    if ei == ej:
                        continue
                    # Right endpoint of edge ej at vertex v
                    right_node = ('R', ej, v)
                    left_to_right[left_node].add(right_node)
        
        # Now run Hopcroft-Karp
        # Map left and right nodes to indices
        left_idx = {node: i for i, node in enumerate(left_nodes)}
        right_idx = {node: i for i, node in enumerate(right_nodes)}
        
        num_left = len(left_nodes)
        num_right = len(right_nodes)
        
        # Build adjacency list for left nodes
        adj_left = [[] for _ in range(num_left)]
        for left_node, right_set in left_to_right.items():
            li = left_idx[left_node]
            for right_node in right_set:
                ri = right_idx[right_node]
                adj_left[li].append(ri)
        
        # Hopcroft-Karp algorithm
        # pairU[i] = right node matched to left node i, or -1
        # pairV[j] = left node matched to right node j, or -1
        # dist[i] = distance from left node i in the BFS tree
        pairU = [-1] * num_left
        pairV = [-1] * num_right
        dist = [0] * num_left
        
        def bfs():
            queue = deque()
            found = False
            for u in range(num_left):
                if pairU[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = float('inf')
            
            while queue:
                u = queue.popleft()
                for v in adj_left[u]:
                    if pairV[v] != -1:
                        if dist[pairV[v]] == float('inf'):
                            dist[pairV[v]] = dist[u] + 1
                            queue.append(pairV[v])
                    else:
                        found = True
            
            return found
        
        def dfs(u):
            for v in adj_left[u]:
                if pairV[v] == -1 or (dist[pairV[v]] == dist[u] + 1 and dfs(pairV[v])):
                    pairU[u] = v
                    pairV[v] = u
                    return True
            dist[u] = float('inf')
            return False
        
        matching_size = 0
        while bfs():
            for u in range(num_left):
                if pairU[u] == -1:
                    if dfs(u):
                        matching_size += 1
        
        # Reconstruct paths from the matching
        # The matching tells us which stubs are connected
        # pairU[left_idx] = right_idx means the left stub is matched to the right stub
        
        # Build a mapping from (edge_idx, vertex) to what it's matched to
        # For each left node (ei, v), if matched to right node (ej, v), then edge ei and ej are consecutive at vertex v
        
        # We need to build paths. Start from unmatched left nodes (these are path starts)
        # and follow the matching.
        
        # Actually, let's think about this differently.
        # Each directed edge (u,v) can be seen as a stub.
        # If left (ei, u) is matched to right (ej, u), that means the stub of edge ei at vertex u is connected to the stub of edge ej at vertex u.
        # This means that in the path, edge ei and edge ej share vertex u, and they are consecutive.
        
        # To build paths:
        # 1. For each edge, determine its two "connections" (which other edges it's connected to at each endpoint)
        # 2. Build a graph where edges are nodes, and connections are edges
        # 3. Find paths in this graph
        
        # For each edge ei = (u, v):
        #   At vertex u: check if left (ei, u) is matched. If so, to which right node?
        #   At vertex v: check if left (ei, v) is matched. If so, to which right node?
        
        # Let's build a directed graph of edge transitions
        # node: edge_idx
        # edge from ei to ej if ei is connected to ej at some vertex
        
        transitions = defaultdict(list)
        for left_node in left_nodes:
            li = left_idx[left_node]
            if pairU[li] != -1:
                ri = pairU[li]
                right_node = right_nodes[ri]
                # left_node = ('L', ei, v), right_node = ('R', ej, v)
                ei = left_node[1]
                ej = right_node[1]
                transitions[ei].append(ej)
        
        # Now find all paths in the transition graph
        # Each edge can appear at most once in a path (since each stub can only be matched once)
        # Actually, each edge has two stubs, each of which can be matched at most once.
        # So an edge can be in the middle of a path (both stubs matched), at the start (one stub matched), or alone (no stubs matched).
        
        # Let's find paths by following transitions
        visited_edges = set()
        paths = []
        
        # Find all edges that are not "entered" from another edge (i.e., path starts)
        # An edge ei is a path start if it has no incoming transition
        incoming = defaultdict(int)
        for ei, ejs in transitions.items():
            for ej in ejs:
                incoming[ej] += 1
        
        for ei in range(num_edges):
            if ei in visited_edges:
                continue
            if incoming[ei] > 0:
                continue  # Not a start
            
            # Follow the path from ei
            path_edges = []
            current = ei
            while current is not None and current not in visited_edges:
                path_edges.append(current)
                visited_edges.add(current)
                # Find the next edge
                next_edges = transitions.get(current, [])
                if next_edges:
                    current = next_edges[0]
                else:
                    current = None
            
            # Convert edge indices to vertex paths
            if path_edges:
                # Build the vertex sequence
                vertex_path = []
                for i, eidx in enumerate(path_edges):
                    u, v = edge_list[eidx]
                    if i == 0:
                        vertex_path.append(u)
                        vertex_path.append(v)
                    else:
                        # The last vertex of the previous edge should match one endpoint of this edge
                        prev_u, prev_v = edge_list[path_edges[i-1]]
                        if vertex_path[-1] == u:
                            vertex_path.append(v)
                        elif vertex_path[-1] == v:
                            vertex_path.append(u)
                        else:
                            # This shouldn't happen if matching is correct
                            vertex_path.append(u)
                            vertex_path.append(v)
                paths.append(vertex_path)
        
        # Add any remaining edges as single-edge paths
        for ei in range(num_edges):
            if ei not in visited_edges:
                u, v = edge_list[ei]
                paths.append([u, v])
        
        return paths, matching_size

    def merge_paths(paths, edge_set, time_limit=20.0):
        """Aggressively merge paths that share endpoints to reduce count."""
        changed = True
        iterations = 0
        max_iterations = 300

        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            if time.time() - start_time > time_limit:
                break

            # Build endpoint index
            endpoint_map = defaultdict(list)
            for i, path in enumerate(paths):
                endpoint_map[path[0]].append((i, 0))
                if len(path) > 1:
                    endpoint_map[path[-1]].append((i, 1))

            merged_any = False
            new_paths = list(paths)

            for vertex in sorted(endpoint_map.keys()):
                if time.time() - start_time > time_limit:
                    break
                entries = endpoint_map[vertex]
                if len(entries) < 2:
                    continue

                # Try all pairs at this vertex
                for a in range(len(entries)):
                    if merged_any:
                        break
                    for b in range(a + 1, len(entries)):
                        i, ei = entries[a]
                        j, ej = entries[b]
                        if i == j or i >= len(new_paths) or j >= len(new_paths):
                            continue

                        pi = new_paths[i]
                        pj = new_paths[j]

                        merged_path = None
                        if ei == 1 and ej == 0 and pi[-1] == vertex and pj[0] == vertex:
                            merged_path = pi + pj[1:]
                        elif ei == 0 and ej == 1 and pi[0] == vertex and pj[-1] == vertex:
                            merged_path = pj + pi[1:]
                        elif ei == 1 and ej == 1 and pi[-1] == vertex and pj[-1] == vertex:
                            merged_path = pi + pj[::-1][1:]
                        elif ei == 0 and ej == 0 and pi[0] == vertex and pj[0] == vertex:
                            merged_path = pi + pj[::-1][1:]

                        if merged_path is None:
                            continue

                        # Verify simplicity and edge validity
                        if len(merged_path) != len(set(merged_path)):
                            continue
                        valid = True
                        for k in range(len(merged_path) - 1):
                            u, v = merged_path[k], merged_path[k + 1]
                            if tuple(sorted((u, v))) not in edge_set:
                                valid = False
                                break
                        if not valid:
                            continue

                        # Perform merge
                        remove_idx = sorted([i, j], reverse=True)
                        for idx in remove_idx:
                            new_paths.pop(idx)
                        new_paths.append(merged_path)
                        merged_any = True
                        changed = True
                        break
                if merged_any:
                    break

            paths = new_paths

        return paths

    def local_search_split_remerge(paths, edge_set, adj, time_limit=25.0, rng_seed=42):
        """Local search: try splitting paths and remerging to reduce total count."""
        rng = random.Random(rng_seed)
        current_paths = list(paths)
        current_count = len(current_paths)
        best_paths = list(current_paths)
        best_count = current_count

        max_iterations = 200
        for it in range(max_iterations):
            if time.time() - start_time > time_limit:
                break
            if len(current_paths) <= 2:
                break

            # Pick a path to split (prefer longer paths for more potential)
            path_indices = list(range(len(current_paths)))
            path_indices.sort(key=lambda i: len(current_paths[i]), reverse=True)
            
            # Try splitting one of the longer paths
            split_candidates = path_indices[:min(5, len(path_indices))]
            chosen_idx = rng.choice(split_candidates)
            path_to_split = current_paths[chosen_idx]

            if len(path_to_split) < 4:
                continue

            # Choose a split point
            split_pos = rng.randint(1, len(path_to_split) - 2)
            part1 = path_to_split[:split_pos + 1]
            part2 = path_to_split[split_pos:]

            # Create new path set with the split
            new_paths = list(current_paths)
            new_paths.pop(chosen_idx)
            new_paths.append(part1)
            new_paths.append(part2)

            # Now try to merge the new pieces with other paths
            new_paths = merge_paths(new_paths, edge_set, time_limit=min(
                time_limit, time.time() - start_time + 5))

            if len(new_paths) < len(current_paths):
                current_paths = new_paths
                current_count = len(current_paths)
                if current_count < best_count:
                    best_count = current_count
                    best_paths = list(current_paths)

        return best_paths

    def try_ilp_refinement(edges, adj, initial_paths, time_limit=15.0):
        """Use ILP to refine path decomposition if tractable."""
        if time.time() - start_time > time_limit:
            return initial_paths

        edge_set = set(tuple(sorted(e)) for e in edges)
        num_edges = len(edge_set)
        nodes = sorted(adj.keys())

        # Generate candidate paths up to length 7
        candidate_paths = []

        def find_paths_from(start, max_len=7):
            found = []

            def dfs(path, visited):
                if len(path) >= 2:
                    found.append(list(path))
                if len(path) >= max_len:
                    return
                last = path[-1]
                for nb in sorted(adj.get(last, [])):
                    if nb not in visited:
                        visited.add(nb)
                        path.append(nb)
                        dfs(path, visited)
                        path.pop()
                        visited.remove(nb)

            dfs([start], {start})
            return found

        for start in nodes:
            candidate_paths.extend(find_paths_from(start))

        # Filter valid paths
        valid_paths = []
        for path in candidate_paths:
            valid = True
            for k in range(len(path) - 1):
                u, v = path[k], path[k + 1]
                if tuple(sorted((u, v))) not in edge_set:
                    valid = False
                    break
            if valid and len(path) >= 2:
                valid_paths.append(path)

        # Deduplicate
        seen = set()
        unique_paths = []
        for p in valid_paths:
            key = tuple(p)
            key_rev = tuple(reversed(p))
            if key not in seen and key_rev not in seen:
                seen.add(key)
                seen.add(key_rev)
                unique_paths.append(p)

        if len(unique_paths) > 3000:
            return initial_paths

        if not unique_paths:
            return initial_paths

        num_paths = len(unique_paths)
        edge_index = {e: i for i, e in enumerate(edge_set)}

        # Incidence matrix
        A = np.zeros((num_paths, num_edges), dtype=np.int8)
        for i, path in enumerate(unique_paths):
            for k in range(len(path) - 1):
                u, v = path[k], path[k + 1]
                e = tuple(sorted((u, v)))
                A[i, edge_index[e]] = 1

        try:
            from scipy.optimize import milp, LinearConstraint, Bounds

            c = np.ones(num_paths)
            lower = np.ones(num_edges)
            upper = np.ones(num_edges)
            constraint = LinearConstraint(A.T, lb=lower, ub=upper)
            bounds = Bounds(lb=0, ub=1)

            result = milp(
                c=c,
                constraints=constraint,
                bounds=bounds,
                integrality=np.ones(num_paths)
            )

            if result.status == 0 and result.x is not None:
                selected = [i for i in range(num_paths) if result.x[i] > 0.5]
                if len(selected) > 0:
                    covered_edges = set()
                    for i in selected:
                        path = unique_paths[i]
                        for k in range(len(path) - 1):
                            u, v = path[k], path[k + 1]
                            covered_edges.add(tuple(sorted((u, v))))
                    if covered_edges == edge_set:
                        ilp_paths = [unique_paths[i] for i in selected]
                        if len(ilp_paths) < len(initial_paths):
                            return ilp_paths
        except Exception:
            pass

        return initial_paths

    results = []
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]

    for n, p, seed in specs:
        if time.time() - start_time > deadline - 15:
            # Emergency fallback
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue

        edges = instance(n, p, seed)
        adj = get_adjacency(edges)
        edge_set = set(tuple(sorted(e)) for e in edges)

        # Use Hopcroft-Karp matching to get an initial decomposition
        matching_paths, matching_size = hopcroft_karp_matching(edges, adj)
        current_paths = matching_paths
        current_count = len(current_paths)

        # Merge pass
        if time.time() - start_time < deadline - 20:
            merged_paths = merge_paths(current_paths, edge_set, time_limit=deadline - 20)
            if len(merged_paths) < current_count:
                current_paths = merged_paths
                current_count = len(current_paths)

        # Local search with split/remerge
        if time.time() - start_time < deadline - 15:
            ls_paths = local_search_split_remerge(current_paths, edge_set, adj,
                                                   time_limit=deadline - 12,
                                                   rng_seed=seed * 1000 + 42)
            if len(ls_paths) < current_count:
                current_paths = ls_paths
                current_count = len(current_paths)

        # Try ILP refinement if we have time
        if time.time() - start_time < deadline - 10:
            refined_paths = try_ilp_refinement(edges, adj, current_paths,
                                               time_limit=deadline - 5)
            if len(refined_paths) < current_count:
                current_paths = refined_paths
                current_count = len(current_paths)

        results.append(current_paths)

    return results
