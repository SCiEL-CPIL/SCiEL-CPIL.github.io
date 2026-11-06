"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    from random import Random

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_cycle(remaining_edges, n):
        """
        Find a cycle in the remaining edges using DFS.
        Returns a list of vertices forming the cycle, or None.
        """
        # Build adjacency from remaining edges
        adj_list = defaultdict(list)
        for (u, v) in remaining_edges:
            adj_list[u].append(v)
            adj_list[v].append(u)

        # Try to find a cycle starting from each vertex with degree >= 2
        for start in range(n):
            if len(adj_list[start]) < 2:
                continue
            # DFS to find a path back to start
            visited = {start}
            parent = {start: None}
            stack = [start]
            found = False
            cycle_path = []

            while stack and not found:
                node = stack.pop()
                for neighbor in adj_list[node]:
                    if neighbor == start and len(parent) >= 2:
                        # Found a cycle: reconstruct path
                        path = []
                        cur = node
                        while cur is not None:
                            path.append(cur)
                            cur = parent[cur]
                        path.append(start)
                        cycle_path = path
                        found = True
                        break
                    if neighbor not in visited:
                        visited.add(neighbor)
                        parent[neighbor] = node
                        stack.append(neighbor)

            if found:
                return cycle_path

        return None

    def decompose(edges, n):
        """
        Decompose edges into cycles and single edges.
        Greedy: repeatedly find a cycle, remove its edges, until no more cycles.
        Remaining edges are single pieces.
        """
        if not edges:
            return []

        remaining = set(edges)
        pieces = []

        # Iteratively find cycles
        while remaining:
            # Check time
            if time.time() > deadline:
                break

            # Find a cycle
            cycle = find_cycle(remaining, n)

            if cycle is None:
                break

            # Remove cycle edges from remaining
            cycle_edges = set()
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                e = tuple(sorted((u, v)))
                if e in remaining:
                    cycle_edges.add(e)
                else:
                    # Edge not in remaining, this shouldn't happen
                    cycle_edges = None
                    break

            if cycle_edges is None or not cycle_edges:
                # Cycle not valid, break
                break

            # Add cycle as a piece
            pieces.append(cycle)

            # Remove cycle edges from remaining
            remaining -= cycle_edges

        # Add remaining edges as single pieces
        for e in sorted(remaining):
            pieces.append(list(e))

        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose(edges, n)
        results.append(pieces)

    return results
# EVOLVE-BLOCK-END
