"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from collections import defaultdict

def construct():
    from random import Random

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        edge_set = set(tuple(sorted(e)) for e in edges)
        if not edge_set:
            return []

        adj = defaultdict(list)
        for (u, v) in edge_set:
            adj[u].append((v, (u, v)))
            adj[v].append((u, (u, v)))

        used_edges = set()
        pieces = []
        remaining = set(edge_set)

        while remaining:
            # Find any cycle using DFS
            # Pick a start with remaining edges
            start = None
            for v in adj:
                if any(e not in used_edges for (_, e) in adj[v]):
                    start = v
                    break
            if start is None:
                break

            # DFS to find a cycle
            # parent tracking for path reconstruction
            visited = {start}
            parent = {start: None}  # vertex -> (prev_vertex, edge)
            cycle = None
            stack = [start]

            while stack and cycle is None:
                u = stack.pop()
                for (v, e) in adj[u]:
                    if e in used_edges:
                        continue
                    if v == start and u != start:
                        # Found a cycle: u -> start
                        # Reconstruct path from u back to start
                        path = [u]
                        cur = u
                        while parent[cur] is not None:
                            cur, _ = parent[cur]
                            path.append(cur)
                        path.reverse()
                        path.append(start)
                        cycle = path
                        break
                    if v not in visited:
                        visited.add(v)
                        parent[v] = (u, e)
                        stack.append(v)

            if cycle:
                # Remove cycle edges
                for i in range(len(cycle) - 1):
                    u, v = cycle[i], cycle[i + 1]
                    e = tuple(sorted((u, v)))
                    if e in remaining:
                        used_edges.add(e)
                        remaining.discard(e)
                pieces.append(cycle[:-1])
            else:
                # No cycle found; take a single edge
                e = next(iter(remaining))
                used_edges.add(e)
                remaining.discard(e)
                pieces.append([e[0], e[1]])

        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        pieces = decompose(edges)
        results.append(pieces)
    return results
# EVOLVE-BLOCK-END
