"""Program that decomposes graphs into few cycles and edges."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque, defaultdict


def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges, n):
        # Build adjacency
        adj = {i: set() for i in range(n)}
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((min(u, v), max(u, v)))

        remaining = set(edge_set)
        pieces = []

        def find_cycles_in_2factor(matching):
            """Given a matching (dict: u -> v), find all cycles of length >= 3."""
            # Build directed graph where each node has out-degree 1
            # matching[u] = v means edge u->v
            cycles = []
            visited = set()
            for u in matching:
                if u in visited:
                    continue
                # Trace the cycle
                path = [u]
                cur = matching[u]
                while cur != u and cur not in visited:
                    path.append(cur)
                    visited.add(cur)
                    if cur not in matching:
                        break
                    cur = matching[cur]
                if cur == u and len(path) >= 3:
                    cycles.append(path)
                    visited.update(path)
            return cycles

        def max_bipartite_matching(bip_adj, left_nodes, right_nodes):
            """Hopcroft-Karp for maximum bipartite matching."""
            pair_u = {u: -1 for u in left_nodes}
            pair_v = {v: -1 for v in right_nodes}
            dist = {}

            def bfs():
                queue = deque()
                found = False
                for u in left_nodes:
                    if pair_u[u] == -1:
                        dist[u] = 0
                        queue.append(u)
                    else:
                        dist[u] = float('inf')
                while queue:
                    u = queue.popleft()
                    for v in bip_adj[u]:
                        pu = pair_v[v]
                        if pu == -1:
                            found = True
                        elif dist[pu] == float('inf'):
                            dist[pu] = dist[u] + 1
                            queue.append(pu)
                return found

            def dfs(u):
                for v in bip_adj[u]:
                    pu = pair_v[v]
                    if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                        pair_u[u] = v
                        pair_v[v] = u
                        return True
                dist[u] = float('inf')
                return False

            matching_size = 0
            while bfs():
                for u in left_nodes:
                    if pair_u[u] == -1:
                        if dfs(u):
                            matching_size += 1

            # Build matching dict: u -> v
            matching = {}
            for u in left_nodes:
                if pair_u[u] != -1:
                    matching[u] = pair_u[u]
            return matching, matching_size

        def find_max_cycle_collection():
            """
            Find a maximum collection of edge-disjoint cycles using the bipartite double cover.
            The idea: create a bipartite graph where left and right copies of each vertex exist.
            An edge (u,v) in the original graph becomes edges (u_L, v_R) and (v_L, u_R).
            A maximum matching in this bipartite graph corresponds to a 2-matching in the original,
            which decomposes into cycles and paths. We extract only the cycles.
            """
            # Build bipartite graph
            left_nodes = list(range(n))
            right_nodes = list(range(n))
            bip_adj = {u: [] for u in left_nodes}
            for u in range(n):
                for v in adj[u]:
                    if (min(u, v), max(u, v)) in remaining:
                        bip_adj[u].append(v)

            # Run Hopcroft-Karp
            matching, _ = max_bipartite_matching(bip_adj, left_nodes, right_nodes)

            # Find cycles in the matching
            cycles = find_cycles_in_2factor(matching)
            return cycles

        def try_extend_cycle(cycle, remaining_edges, adj_map):
            """Try to extend a cycle by finding additional edges that can be inserted."""
            # This is a simple heuristic: try to find chords that can split and recombine
            # For now, just return the cycle as is
            return cycle

        def greedy_cycle_extraction():
            """
            Greedily extract cycles, preferring longer ones.
            Use the bipartite matching approach to find multiple edge-disjoint cycles at once.
            """
            extracted = []
            while True:
                if not remaining:
                    break

                # Find maximum collection of edge-disjoint cycles
                cycles = find_max_cycle_collection()

                if not cycles:
                    # Try to find any single cycle using DFS
                    cycle = find_single_cycle()
                    if cycle is None:
                        break
                    cycles = [cycle]

                # Remove edges of found cycles from remaining
                for cycle in cycles:
                    for i in range(len(cycle)):
                        u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                        e = (min(u, v), max(u, v))
                        remaining.discard(e)
                    extracted.append(cycle)

                # Check for time
                if time.time() > deadline - 5:
                    break

            return extracted

        def find_single_cycle():
            """Find a single cycle using DFS."""
            for start in range(n):
                if len(adj[start]) < 2:
                    continue
                # DFS to find a cycle
                stack = [(start, [start])]
                visited = {start}
                while stack:
                    node, path = stack.pop()
                    for neighbor in sorted(adj[node], reverse=True):
                        e = (min(node, neighbor), max(node, neighbor))
                        if e not in remaining:
                            continue
                        if neighbor == start and len(path) >= 3:
                            return path
                        if neighbor not in visited:
                            visited.add(neighbor)
                            stack.append((neighbor, path + [neighbor]))
            return None

        def find_long_cycle_heuristic():
            """
            Find a longer cycle using a heuristic: start from high-degree vertices
            and try to find cycles through them.
            """
        best_cycle = None
        for start in range(n):
            if len(adj[start]) < 2:
                continue
            # Try to find a cycle starting and ending at start
            # Use a modified DFS that prefers going through high-degree nodes
            visited = set()
            parent = {}
            cycle = None

            def dfs(node, depth):
                nonlocal cycle
                if cycle is not None:
                    return
                if depth > 60:  # limit depth for speed
                    return
                visited.add(node)
                # Sort neighbors by degree descending to prefer high-degree paths
                for neighbor in sorted(adj[node], key=lambda x: len(adj[x]), reverse=True):
                    e = (min(node, neighbor), max(node, neighbor))
                    if e not in remaining:
                        continue
                    if neighbor == start and len(visited) >= 3:
                        # Found a cycle
                        new_cycle = [start]
                        cur = node
                        while cur != start:
                            new_cycle.append(cur)
                            cur = parent[cur]
                        if best_cycle is None or len(new_cycle) > len(best_cycle):
                            best_cycle = new_cycle
                        cycle = new_cycle
                        return
                    if neighbor not in visited:
                        parent[neighbor] = node
                        dfs(neighbor, depth + 1)
                        if cycle is not None:
                            return

            dfs(start, 0)
            if cycle is not None:
                return cycle
        return best_cycle

        # Main decomposition loop
        # First, try to extract as many cycles as possible using the bipartite matching approach
        # Then, handle remaining edges individually

        # Iterative approach: repeatedly find max cycle collections
        max_iterations = 100
        for _ in range(max_iterations):
            if time.time() > deadline - 5:
                break
            if not remaining:
                break

            # Find maximum collection of edge-disjoint cycles
            cycles = find_max_cycle_collection()

            if not cycles:
                # Try to find a single longer cycle
                cycle = find_long_cycle_heuristic()
                if cycle is None:
                    cycle = find_single_cycle()
                if cycle is None:
                    break
                cycles = [cycle]

            # Remove edges of found cycles from remaining
            for cycle in cycles:
                for i in range(len(cycle)):
                    u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                    e = (min(u, v), max(u, v))
                    remaining.discard(e)
                pieces.append(cycle)

        # Remaining edges become individual pieces
        for u, v in sorted(remaining):
            pieces.append([u, v])

        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose(edges, n)
        results.append(pieces)

    return results
# EVOLVE-BLOCK-END
