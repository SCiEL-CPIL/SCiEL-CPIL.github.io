import time
import random
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin for 150s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def extract_2factor(adj, n, remaining_edges):
        """
        Extract a 2-factor (collection of vertex-disjoint cycles) from the graph
        using max-flow. Returns list of cycles (each as list of vertices).
        """
        if not remaining_edges:
            return []

        # Build flow network
        source = n * 2
        sink = n * 2 + 1
        num_nodes = n * 2 + 2

        # Adjacency list for flow network
        # Each edge: [to, cap, rev_index]
        flow_adj = [[] for _ in range(num_nodes)]

        def add_edge(u, v, cap):
            flow_adj[u].append([v, cap, len(flow_adj[v])])
            flow_adj[v].append([u, 0, len(flow_adj[u]) - 1])

        # Source to vertex_in
        for i in range(n):
            add_edge(source, i, 1)  # source -> i (in)

        # vertex_out to sink
        for i in range(n):
            add_edge(i + n, sink, 1)  # i+n (out) -> sink

        # vertex_in -> vertex_out (capacity 1, ensures each vertex used at most once)
        for i in range(n):
            add_edge(i, i + n, 1)

        # For each remaining edge (u,v), add directed edges both ways
        for (u, v) in remaining_edges:
            # u_out -> v_in and v_out -> u_in
            add_edge(u + n, v, 1)
            add_edge(v + n, u, 1)

        # Max flow using Edmonds-Karp (BFS)
        def max_flow():
            total_flow = 0
            while True:
                # BFS to find augmenting path
                parent = [-1] * num_nodes
                parent_edge = [-1] * num_nodes  # index of edge used
                queue = deque([source])
                parent[source] = source
                found = False

                while queue and not found:
                    u = queue.popleft()
                    for idx, (v, cap, _) in enumerate(flow_adj[u]):
                        if cap > 0 and parent[v] == -1:
                            parent[v] = u
                            parent_edge[v] = idx
                            if v == sink:
                                found = True
                                break
                            queue.append(v)

                if not found:
                    break

                # Find bottleneck
                bottleneck = float('inf')
                v = sink
                while v != source:
                    u = parent[v]
                    idx = parent_edge[v]
                    bottleneck = min(bottleneck, flow_adj[u][idx][1])
                    v = u

                # Update residual capacities
                v = sink
                while v != source:
                    u = parent[v]
                    idx = parent_edge[v]
                    flow_adj[u][idx][1] -= bottleneck
                    rev_idx = flow_adj[u][idx][2]
                    flow_adj[v][rev_idx][1] += bottleneck
                    v = u

                total_flow += bottleneck

            return total_flow

        flow = max_flow()
        if flow == 0:
            return []

        # Reconstruct cycles from flow
        next_vertex = [-1] * n
        for i in range(n):
            out_node = i + n
            for idx, (v, cap, rev) in enumerate(flow_adj[out_node]):
                if v < n:  # v is an in-node
                    # Check reverse edge
                    rev_cap = flow_adj[v][rev][1]
                    if rev_cap > 0:
                        # Flow was sent from i to v
                        next_vertex[i] = v
                        break

        # Now find cycles in the permutation next_vertex
        cycles = []
        visited = [False] * n
        for i in range(n):
            if not visited[i] and next_vertex[i] != -1:
                cycle = []
                cur = i
                while not visited[cur]:
                    visited[cur] = True
                    cycle.append(cur)
                    cur = next_vertex[cur]
                if len(cycle) >= 3:
                    cycles.append(cycle)

        return cycles

    def find_longest_cycle(adj, n, remaining_edges):
        """Find the longest cycle using randomized DFS with multiple restarts."""
        best_len = 0
        best_cycle = []
        start_time = time.time()
        rng = random.Random(42)

        candidates = [v for v in range(n) if len(adj[v]) >= 2]
        if not candidates:
            return []

        # Sort by degree descending
        candidates.sort(key=lambda v: -len(adj[v]))

        # Try multiple restarts with randomization
        max_restarts = min(20, len(candidates))
        for r in range(max_restarts):
            if time.time() > deadline or time.time() - start_time > 5.0:
                break

            # Pick a start vertex: mix of high-degree and random
            if r < max_restarts // 3:
                start = candidates[r % len(candidates)]
            else:
                start = rng.choice(candidates)

            # DFS with backtracking to find long cycles
            # Use iterative DFS to avoid recursion limit
            visited = {start}
            stack = [(start, [start], None)]  # (node, path, parent)
            found_any = False

            while stack and time.time() < deadline and time.time() - start_time < 5.0:
                node, path, parent = stack.pop()

                # Check if we can close a cycle back to start
                if node == start and len(path) >= 3:
                    if len(path) > best_len:
                        best_len = len(path)
                        best_cycle = path[:]
                    found_any = True
                    continue

                neighbors = list(adj[node])
                # Prioritize: unvisited first, then high degree
                neighbors.sort(key=lambda x: (x in visited, -len(adj[x])))

                for nb in reversed(neighbors):  # Reverse so first in original order is processed first
                    e = (min(node, nb), max(node, nb))
                    if e not in remaining_edges:
                        continue

                    if nb == start and len(path) >= 3:
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = path[:]
                        found_any = True
                    elif nb not in visited:
                        new_visited = visited | {nb}
                        stack.append((nb, path + [nb], node))

            if found_any and best_len >= n * 0.6:
                break  # Good enough, stop early

        return best_cycle

    def find_path_bfs(start, end, adj, remaining_edges, exclude_edge=None):
        """Find a path from start to end using remaining edges via BFS."""
        if start == end:
            return None
        queue = deque([(start, [start])])
        visited = {start}

        while queue and time.time() < deadline:
            node, path = queue.popleft()
            for neighbor in sorted(adj[node]):
                e = (min(node, neighbor), max(node, neighbor))
                if e not in remaining_edges:
                    continue
                if e == exclude_edge:
                    continue
                if neighbor == end:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    def try_merge_edges_into_cycles(pieces, n, adj, remaining_edges):
        """Try to merge single edges into cycles by finding paths between their endpoints."""
        improved = False
        i = 0
        while i < len(pieces) and time.time() < deadline:
            if len(pieces[i]) == 2:
                u, v = pieces[i]
                e = (min(u, v), max(u, v))
                if e in remaining_edges:
                    path = find_path_bfs(u, v, adj, remaining_edges, exclude_edge=e)
                    if path and len(path) >= 3:
                        # Replace single edge with cycle
                        pieces[i] = path
                        for j in range(len(path)):
                            a, b = path[j], path[(j + 1) % len(path)]
                            ee = (min(a, b), max(a, b))
                            if ee in remaining_edges:
                                remaining_edges.discard(ee)
                                adj[a].discard(b)
                                adj[b].discard(a)
                        improved = True
                    else:
                        i += 1
                else:
                    i += 1
            else:
                i += 1
        return improved

    def try_merge_two_edges(pieces, n, adj, remaining_edges):
        """Try to merge pairs of single edges into a 4-cycle or longer cycle."""
        improved = False
        i = 0
        while i < len(pieces) and time.time() < deadline:
            if len(pieces[i]) == 2:
                u1, v1 = pieces[i]
                e1 = (min(u1, v1), max(u1, v1))
                if e1 in remaining_edges:
                    j = i + 1
                    while j < len(pieces) and time.time() < deadline:
                        if len(pieces[j]) == 2:
                            u2, v2 = pieces[j]
                            e2 = (min(u2, v2), max(u2, v2))
                            if e2 in remaining_edges and e2 != e1:
                                path1 = find_path_bfs(v1, u2, adj, remaining_edges, exclude_edge=e1)
                                if path1 and len(path1) >= 2:
                                    path2 = find_path_bfs(v2, u1, adj, remaining_edges, exclude_edge=e2)
                                    if path2 and len(path2) >= 2:
                                        internal1 = set(path1[1:-1])
                                        internal2 = set(path2[1:-1])
                                        if not (internal1 & internal2):
                                            cycle = [u1, v1] + path1[1:] + [v2] + path2[1:]
                                            if len(set(cycle)) == len(cycle) and len(cycle) >= 4:
                                                valid = True
                                                for k in range(len(cycle)):
                                                    a, b = cycle[k], cycle[(k + 1) % len(cycle)]
                                                    ee = (min(a, b), max(a, b))
                                                    if ee not in remaining_edges:
                                                        valid = False
                                                        break
                                                if valid:
                                                    pieces[i] = cycle
                                                    del pieces[j]
                                                    for k in range(len(cycle)):
                                                        a, b = cycle[k], cycle[(k + 1) % len(cycle)]
                                                        ee = (min(a, b), max(a, b))
                                                        if ee in remaining_edges:
                                                            remaining_edges.discard(ee)
                                                            adj[a].discard(b)
                                                            adj[b].discard(a)
                                                    improved = True
                                                    break
                        j += 1
                    if improved:
                        continue
                i += 1
            else:
                i += 1
        return improved

    def decompose_one_graph(spec):
        """Decompose a single graph into cycles and edges."""
        edges = instance(*spec)
        n = spec[0]
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        pieces = []
        remaining_edges = set(edges)

        # Phase 1: Extract 2-factors greedily
        while remaining_edges and time.time() < deadline:
            cycles = extract_2factor(adj, n, remaining_edges)
            if cycles:
                for cycle in cycles:
                    pieces.append(cycle)
                    for i in range(len(cycle)):
                        u = cycle[i]
                        v = cycle[(i + 1) % len(cycle)]
                        e = (min(u, v), max(u, v))
                        if e in remaining_edges:
                            remaining_edges.discard(e)
                            adj[u].discard(v)
                            adj[v].discard(u)
            else:
                # No more 2-factors; try longest cycle
                best_cycle = find_longest_cycle(adj, n, remaining_edges)
                if best_cycle:
                    pieces.append(best_cycle)
                    for i in range(len(best_cycle)):
                        u = best_cycle[i]
                        v = best_cycle[(i + 1) % len(cycle) if False else (i + 1) % len(best_cycle)]
                        e = (min(u, v), max(u, v))
                        if e in remaining_edges:
                            remaining_edges.discard(e)
                            adj[u].discard(v)
                            adj[v].discard(u)
                else:
                    # No cycles left; add remaining edges as single pieces
                    for e in remaining_edges:
                        pieces.append(list(e))
                    break

        # Phase 2: Post-processing - merge single edges into cycles
        if time.time() < deadline:
            try_merge_edges_into_cycles(pieces, n, adj, remaining_edges)

        # Phase 3: Try to merge pairs of single edges
        if time.time() < deadline:
            try_merge_two_edges(pieces, n, adj, remaining_edges)

        return pieces

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))

    for spec in specs:
        if time.time() > deadline:
            # Fallback
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        results.append(decompose_one_graph(spec))

    return results
