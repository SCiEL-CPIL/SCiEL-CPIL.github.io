"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Generate the three fixed instance graphs
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
        """Decompose edges into cycles and single edges, minimizing piece count."""
        if not edges:
            return []

        # Build adjacency structure
        adj = defaultdict(set)
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v))

        remaining_edges = set(edge_set)
        pieces = []

        # Helper to find a cycle using DFS with backtracking
        def find_cycle(start, remaining):
            """Find a cycle starting and ending at start using only edges in remaining."""
            # Try to find a cycle by DFS
            visited = {start}
            path = [start]
            
            def dfs(node):
                if len(path) >= 3 and node == start:
                    return True
                for neighbor in sorted(adj[node]):
                    if (min(node, neighbor), max(node, neighbor)) not in remaining:
                        continue
                    if neighbor == start and len(path) >= 3:
                        return True
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
                # Extract the cycle
                cycle_nodes = path[:]
                # Remove start from end if duplicated
                if cycle_nodes[0] == cycle_nodes[-1]:
                    cycle_nodes = cycle_nodes[:-1]
                return cycle_nodes
            return None

        # Greedy approach: repeatedly find cycles, prefer longer ones
        # Use a heuristic: start from vertices with high degree in remaining graph
        max_iterations = 10000
        iteration = 0

        while remaining_edges and iteration < max_iterations:
            if time.time() > time_limit:
                break
            iteration += 1

            # Find all possible cycles and pick the longest
            best_cycle = None
            best_len = 0

            # Try starting from each vertex with remaining edges
            vertices_with_edges = set()
            for u, v in remaining_edges:
                vertices_with_edges.add(u)
                vertices_with_edges.add(v)

            # Sort by degree in remaining graph for better heuristic
            deg = defaultdict(int)
            for u, v in remaining_edges:
                deg[u] += 1
                deg[v] += 1

            for start in sorted(vertices_with_edges, key=lambda x: -deg[x]):
                cycle = find_cycle(start, remaining_edges)
                if cycle and len(cycle) > best_len:
                    best_len = len(cycle)
                    best_cycle = cycle
                    if best_len >= 10:  # Good enough, stop searching
                        break

            if best_cycle:
                # Add the cycle as a piece
                pieces.append(best_cycle)
                # Remove the edges of this cycle from remaining
                n = len(best_cycle)
                for i in range(n):
                    u = best_cycle[i]
                    v = best_cycle[(i + 1) % n]
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

        return pieces

    results = []
    for i, edges in enumerate(graphs):
        # Allocate time based on remaining time
        remaining_time = deadline - time.time()
        time_per_graph = remaining_time / (len(graphs) - i)
        pieces = decompose(edges, time.time() + time_per_graph)
        results.append(pieces)

    return results
# EVOLVE-BLOCK-END
