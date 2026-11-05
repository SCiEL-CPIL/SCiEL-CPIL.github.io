"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        """Decompose edge set into cycles and single edges, minimizing piece count."""
        if not edges:
            return []

        # Build adjacency
        adj = defaultdict(list)
        edge_set = set()
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            edge_set.add((u, v))
            edge_set.add((v, u))

        pieces = []
        remaining_edges = set(edges)

        # Try to find cycles greedily
        while remaining_edges:
            if time.time() > deadline:
                break

            # Find a cycle using DFS
            cycle = None
            start = None
            for edge in list(remaining_edges):
                u, v = edge
                # Try to find a cycle starting from u
                path = [u]
                visited = {u}
                stack = [(u, iter(sorted(adj[u])))]
                found = False
                while stack and not found:
                    node, neighbors = stack[-1]
                    for nxt in neighbors:
                        if nxt == path[0] and len(path) >= 3 and (node, nxt) in remaining_edges:
                            # Found a cycle
                            cycle = path + [nxt]
                            found = True
                            break
                        if nxt not in visited and (node, nxt) in remaining_edges:
                            visited.add(nxt)
                            path.append(nxt)
                            stack.append((nxt, iter(sorted(adj[nxt]))))
                            break
                    if not found:
                        stack.pop()
                        if path:
                            path.pop()
                if found:
                    break

            if cycle:
                # Extract the cycle edges
                cycle_edges = set()
                for i in range(len(cycle) - 1):
                    a, b = cycle[i], cycle[i + 1]
                    e = tuple(sorted((a, b)))
                    cycle_edges.add(e)
                pieces.append(cycle[:-1])  # closing edge is implied
                remaining_edges -= cycle_edges
            else:
                # No more cycles; add remaining edges as single-edge pieces
                for e in remaining_edges:
                    pieces.append(list(e))
                break

        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        pieces = decompose(edges)
        results.append(pieces)

    return results
# EVOLVE-BLOCK-END
