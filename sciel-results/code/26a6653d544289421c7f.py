import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    graphs = [instance(*spec) for spec in specs]

    def decompose(edges, time_limit):
        if not edges:
            return []

        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        remaining_edges = set((min(u, v), max(u, v)) for u, v in edges)
        pieces = []

        def find_cycle(start, remaining):
            """Find a cycle starting and ending at start using only edges in remaining."""
            visited = {start}
            path = [start]

            def dfs(node):
                for neighbor in sorted(adj[node]):
                    e = (min(node, neighbor), max(node, neighbor))
                    if e not in remaining:
                        continue
                    if neighbor == start:
                        if len(path) >= 3:
                            return True
                        continue
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    path.append(neighbor)
                    if dfs(neighbor):
                        return True
                    path.pop()
                    visited.discard(neighbor)
                return False

            if dfs(start):
                cycle_nodes = path[:]
                if cycle_nodes[0] == cycle_nodes[-1]:
                    cycle_nodes = cycle_nodes[:-1]
                return cycle_nodes
            return None

        def find_best_cycle(remaining):
            """Find the longest cycle in the remaining graph."""
            best_cycle = None
            best_len = 0

            vertices_with_edges = set()
            for u, v in remaining:
                vertices_with_edges.add(u)
                vertices_with_edges.add(v)

            deg = defaultdict(int)
            for u, v in remaining:
                deg[u] += 1
                deg[v] += 1

            # Try from high-degree vertices first
            for start in sorted(vertices_with_edges, key=lambda x: -deg[x]):
                cycle = find_cycle(start, remaining)
                if cycle and len(cycle) > best_len:
                    best_len = len(cycle)
                    best_cycle = cycle
                    if best_len >= 8:
                        break

            return best_cycle

        def try_improve(pieces, remaining, time_limit, max_rounds=50):
            """Try to merge single edges into cycles to reduce piece count."""
            improved = True
            rounds = 0
            while improved and rounds < max_rounds and time.time() < time_limit:
                improved = False
                rounds += 1

                # Find all single-edge pieces
                single_edges = []
                for piece in pieces:
                    if len(piece) == 2:
                        single_edges.append(piece)

                if len(single_edges) < 2:
                    break

                # Try to find cycles that include some single edges
                # We'll look for cycles in the full graph that use at least 2 single edges
                # and replace those single edges with the cycle

                # Build a set of single edge endpoints
                single_set = set()
                for u, v in single_edges:
                    single_set.add((min(u, v), max(u, v)))

                # Try to find cycles that use multiple single edges
                # This is complex; instead, let's try a simpler approach:
                # For each pair of single edges, see if they can be part of a cycle
                # that also uses some cycle edges

                # Alternative: just try to find any cycle in the remaining graph
                # that wasn't found before (maybe we missed it)
                cycle = find_best_cycle(remaining)
                if cycle:
                    n = len(cycle)
                    edges_to_remove = []
                    for i in range(n):
                        u = cycle[i]
                        v = cycle[(i + 1) % n]
                        e = (min(u, v), max(u, v))
                        edges_to_remove.append(e)

                    # Check if all these edges are in remaining
                    all_in_remaining = all(e in remaining for e in edges_to_remove)
                    if all_in_remaining:
                        # Add cycle, remove edges
                        pieces.append(cycle)
                        for e in edges_to_remove:
                            remaining.discard(e)
                        improved = True

                # Also try to find cycles that replace single edges
                # Look for paths between single edge endpoints
                for i in range(len(single_edges)):
                    if time.time() > time_limit:
                        break
                    u1, v1 = single_edges[i]
                    for j in range(i + 1, len(single_edges)):
                        if time.time() > time_limit:
                            break
                        u2, v2 = single_edges[j]

                        # Try to find a path from v1 to u2 that doesn't use single edges
                        # and a path from v2 to u1, forming a cycle
                        # This is getting complex; skip for now

            return pieces, remaining

        # Main greedy loop
        max_iterations = 10000
        iteration = 0

        while remaining_edges and iteration < max_iterations:
            if time.time() > time_limit:
                break
            iteration += 1

            cycle = find_best_cycle(remaining_edges)
            if cycle:
                pieces.append(cycle)
                n = len(cycle)
                for i in range(n):
                    u = cycle[i]
                    v = cycle[(i + 1) % n]
                    e = (min(u, v), max(u, v))
                    if e in remaining_edges:
                        remaining_edges.discard(e)
            else:
                # No more cycles, add all remaining edges as single edges
                for u, v in sorted(remaining_edges):
                    pieces.append([u, v])
                remaining_edges.clear()
                break

        # Any leftover edges become single-edge pieces
        for u, v in sorted(remaining_edges):
            pieces.append([u, v])

        # Try to improve by finding more cycles
        if time.time() < time_limit - 5:
            pieces, remaining_edges = try_improve(pieces, remaining_edges, time_limit)
            for u, v in sorted(remaining_edges):
                pieces.append([u, v])

        return pieces

    results = []
    for i, edges in enumerate(graphs):
        remaining_time = deadline - time.time()
        time_per_graph = remaining_time / (len(graphs) - i)
        pieces = decompose(edges, time.time() + time_per_graph)
        results.append(pieces)

    return results
