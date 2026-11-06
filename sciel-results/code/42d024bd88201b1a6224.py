import time
import sys
from random import Random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    TIME_LIMIT = 140.0

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def hopcroft_karp(n_left, n_right, adj):
        """
        Finds maximum matching in a bipartite graph.
        n_left, n_right: number of vertices on left and right.
        adj: list of lists, adj[u] = list of v in right.
        Returns:
        match_left: list where match_left[u] = v or -1
        match_right: list where match_right[v] = u or -1
        """
        INF = float('inf')
        match_left = [-1] * n_left
        match_right = [-1] * n_right
        dist = [0] * n_left

        def bfs():
            queue = deque()
            for u in range(n_left):
                if match_left[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = INF
            found = False
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    mu = match_right[v]
                    if mu == -1:
                        found = True
                    elif dist[mu] == INF:
                        dist[mu] = dist[u] + 1
                        queue.append(mu)
            return found

        def dfs(u):
            for v in adj[u]:
                mu = match_right[v]
                if mu == -1 or (dist[mu] == dist[u] + 1 and dfs(mu)):
                    match_left[u] = v
                    match_right[v] = u
                    return True
            dist[u] = INF
            return False

        while bfs():
            for u in range(n_left):
                if match_left[u] == -1:
                    dfs(u)
        
        return match_left, match_right

    def find_2_matching_cycles(vertices, edges_set):
        """
        Given a set of vertices and a set of edges, find a maximum 2-matching
        and return the list of cycles found within that 2-matching.
        """
        if not edges_set:
            return []
        
        # Build bipartite double cover
        # Left: u_out, Right: v_in
        # Edge (u,v) in original -> (u_out, v_in) and (v_out, u_in) in bipartite
        
        # We only consider vertices that are in edges_set
        active_vertices = set()
        for u, v in edges_set:
            active_vertices.add(u)
            active_vertices.add(v)
        
        if not active_vertices:
            return []
            
        # Map vertices to indices 0..N-1 for the bipartite graph
        # Left side: 0..N-1, Right side: N..2N-1
        v_list = sorted(active_vertices)
        v_index = {v: i for i, v in enumerate(v_list)}
        N = len(v_list)
        
        adj = [[] for _ in range(N)]
        
        for u, v in edges_set:
            ui = v_index[u]
            vi = v_index[v]
            # u_out -> v_in
            adj[ui].append(vi)
            # v_out -> u_in
            adj[vi].append(ui)
            
        match_left, match_right = hopcroft_karp(N, N, adj)
        
        # Reconstruct the 2-matching edges in the original graph
        # match_left[u] = v means edge (u, v) is in the matching?
        # In the bipartite graph, an edge from u_out to v_in corresponds to original edge (u, v).
        # An edge from v_out to u_in corresponds to original edge (v, u) which is (u, v).
        
        # We need to collect the edges (u, v) such that either:
        # 1. match_left[u] == v (edge u_out->v_in exists)
        # 2. match_left[v] == u (edge v_out->u_in exists)
        
        # Actually, a 2-matching edge (u,v) is present if:
        # - u_out is matched to v_in (match_left[u] == v)
        # - OR v_out is matched to u_in (match_left[v] == u)
        # Note: It's possible both are true? 
        # If match_left[u] == v, then u_out is used.
        # If match_left[v] == u, then v_out is used.
        # These are distinct nodes in the bipartite graph.
        # So an original edge (u,v) is in the 2-matching if at least one of the corresponding bipartite edges is in the matching.
        
        # However, we must be careful. A 2-matching requires degree <= 2 at each vertex.
        # The matching ensures that each u_out has at most 1 outgoing edge, and each u_in has at most 1 incoming edge.
        # So the degree of u in the 2-matching is (1 if u_out matched) + (1 if u_in matched).
        # This is exactly the definition of a 2-matching.
        
        two_matching_edges = set()
        for u in range(N):
            v = match_left[u]
            if v != -1:
                orig_u = v_list[u]
                orig_v = v_list[v]
                two_matching_edges.add((min(orig_u, orig_v), max(orig_u, orig_v)))
        
        # Now find cycles in the subgraph formed by two_matching_edges
        # The subgraph has max degree 2. Components are paths or cycles.
        # We only want cycles.
        
        adj_2m = defaultdict(set)
        for u, v in two_matching_edges:
            adj_2m[u].add(v)
            adj_2m[v].add(u)
            
        visited_nodes = set()
        cycles = []
        
        for node in v_list:
            if node in visited_nodes:
                continue
            # Traverse the component
            if len(adj_2m[node]) < 2:
                # It's a path or isolated, skip (or handle later, but we only extract cycles)
                # Mark as visited to avoid re-processing
                # We can do a DFS to mark the whole path as visited
                stack = [node]
                while stack:
                    n = stack.pop()
                    if n in visited_nodes:
                        continue
                    visited_nodes.add(n)
                    for nb in adj_2m[n]:
                        if nb not in visited_nodes:
                            stack.append(nb)
                continue
            
            # It's part of a cycle
            # Find the cycle
            cycle = []
            curr = node
            prev = -1
            while True:
                cycle.append(curr)
                visited_nodes.add(curr)
                # Find next node
                # curr has 2 neighbors in the 2-matching
                nbs = [n for n in adj_2m[curr] if n != prev]
                if not nbs:
                    break
                next_node = nbs[0]
                if next_node == node:
                    break
                prev, curr = curr, next_node
                if curr == node:
                    break
            
            if len(cycle) >= 3 and cycle[0] == cycle[-1] if cycle else False:
                 # The loop above might not close properly in all cases
                 pass

        # Better cycle extraction for max degree 2 graph
        visited_nodes = set()
        cycles = []
        
        for start in v_list:
            if start in visited_nodes:
                continue
            if len(adj_2m[start]) != 2:
                # Part of a path, mark the whole path as visited
                stack = [start]
                while stack:
                    n = stack.pop()
                    if n in visited_nodes:
                        continue
                    visited_nodes.add(n)
                    for nb in adj_2m[n]:
                        if nb not in visited_nodes:
                            stack.append(nb)
                continue
            
            # Find cycle starting from start
            cycle = [start]
            visited_nodes.add(start)
            curr = start
            prev = -1
            while True:
                nbs = [n for n in adj_2m[curr] if n != prev]
                if not nbs:
                    break
                next_node = nbs[0]
                if next_node == start:
                    break
                cycle.append(next_node)
                visited_nodes.add(next_node)
                prev, curr = curr, next_node
                
            if len(cycle) >= 3:
                # Verify it's a cycle
                # The path is start -> ... -> curr, and curr is connected to start
                # We need to ensure all edges exist
                valid = True
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i+1)%len(cycle)]
                    if (min(u,v), max(u,v)) not in two_matching_edges:
                        valid = False
                        break
                if valid:
                    cycles.append(cycle)
                    
        return cycles

    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []

    for spec in specs:
        if time.time() - start_time > TIME_LIMIT:
            results.append([list(e) for e in instance(*spec)])
            continue

        edges = instance(*spec)
        if not edges:
            results.append([])
            continue

        remaining_edges = set(edges)
        pieces = []
        
        # Iteratively extract cycles using 2-matching
        # To avoid infinite loops or getting stuck, we limit iterations
        max_iters = 100
        
        for _ in range(max_iters):
            if time.time() - start_time > TIME_LIMIT:
                for e in remaining_edges:
                    pieces.append(list(e))
                break
                
            if not remaining_edges:
                break
                
            cycles = find_2_matching_cycles(list(range(100)), remaining_edges)
            
            if not cycles:
                # No cycles found in the 2-matching of the remaining graph
                # This implies the remaining graph is a forest? Or the 2-matching only found paths.
                # If the graph has cycles, a max 2-matching should find some.
                # If no cycles are found, we can just add all remaining edges as single edges.
                for e in remaining_edges:
                    pieces.append(list(e))
                break
            
            # Add cycles to pieces and remove their edges
            for cycle in cycles:
                pieces.append(cycle)
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    e = (min(u, v), max(u, v))
                    remaining_edges.discard(e)
        
        if remaining_edges:
            # Fallback for any edges left (shouldn't happen if we break correctly)
            for e in remaining_edges:
                pieces.append(list(e))
                
        results.append(pieces)

    return results
