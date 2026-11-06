"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 140.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_graph(n, edges):
        remaining = set(edges)
        pieces = []

        while remaining and time.time() < deadline - 5:
            # Build adjacency from remaining edges
            adj = defaultdict(set)
            for u, v in remaining:
                adj[u].add(v)
                adj[v].add(u)

            # Find a cycle using DFS
            cycle = None
            for start in range(n):
                if not adj[start]:
                    continue
                # Try to find a cycle starting from 'start'
                visited = set()
                path = []

                def dfs(node, parent):
                    nonlocal cycle
                    visited.add(node)
                    path.append(node)
                    for nbr in sorted(adj[node]):
                        if (min(node, nbr), max(node, nbr)) not in remaining:
                            continue
                        if nbr == parent:
                            continue
                        if nbr == start and len(path) >= 3:
                            cycle = path[:]
                            return True
                        if nbr not in visited:
                            if dfs(nbr, node):
                                return True
                    path.pop()
                    return False

                visited = set()
                path = []
                if dfs(start, -1):
                    break

            if cycle and len(cycle) >= 3:
                # Remove cycle edges from remaining
                for i in range(len(cycle) - 1):
                    u, v = cycle[i], cycle[i + 1]
                    edge = (min(u, v), max(u, v))
                    remaining.discard(edge)
                pieces.append(cycle)
            else:
                # No cycle found; add all remaining as edges
                for u, v in remaining:
                    pieces.append([u, v])
                remaining = set()
                break

        # Add any remaining edges
        for u, v in remaining:
            pieces.append([u, v])

        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose_graph(n, edges)
        results.append(pieces)

    return results
# EVOLVE-BLOCK-END
