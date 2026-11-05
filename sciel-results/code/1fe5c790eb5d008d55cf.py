"""Decompose three fixed Erdős-Gallai instance graphs into few cycles and edges."""
import random
from random import Random

def construct():
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edge_list):
        # Build adjacency and edge set
        edge_set = {tuple(sorted(e)) for e in edge_list}
        n = max(max(e) for e in edge_set) + 1 if edge_set else 0
        adj = {i: set() for i in range(n)}
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)

        pieces = []

        def remove_cycle(cycle):
            nonlocal edge_set, adj
            m = len(cycle)
            for i in range(m):
                a = cycle[i]
                b = cycle[(i + 1) % m]
                key = (min(a, b), max(a, b))
                if key in edge_set:
                    edge_set.discard(key)
                    adj[a].discard(b)
                    adj[b].discard(a)

        # Try to find a long simple cycle using randomized DFS
        def find_long_cycle(rng, max_attempts=200, max_depth=50):
            best_cycle = None
            for _ in range(max_attempts):
                start = rng.choice(list(range(n)))
                if len(adj[start]) < 2:
                    continue
                # DFS to find a cycle back to start
                visited = {start}
                path = [start]
                # Use a stack for DFS with backtracking
                stack = [(start, list(adj[start]))]
                found = False
                # Simple recursive-style DFS with limit
                def dfs(node, path, visited):
                    nonlocal found, best_cycle
                    if found:
                        return
                    for nb in list(adj[node]):
                        if nb == start and len(path) >= 3:
                            # Found a cycle
                            cycle = path[:]
                            if best_cycle is None or len(cycle) > len(best_cycle):
                                best_cycle = cycle
                            found = True
                            return
                        if nb not in visited:
                            visited.add(nb)
                            path.append(nb)
                            dfs(nb, path, visited)
                            path.pop()
                            visited.discard(nb)
                            if found:
                                return
                # Limit search depth
                dfs(start, path, visited)
                if found and best_cycle is not None:
                    # Keep looking for longer
                    continue
            return best_cycle

        # Main loop: extract cycles
        for _ in range(200):  # safety limit
            if not edge_set:
                break
            # Check if any cycle exists
            has_cycle = False
            for v in range(n):
                if len(adj[v]) >= 2:
                    has_cycle = True
                    break
            if not has_cycle:
                break

            # Try to find a long cycle with several random seeds
            best = None
            for trial in range(30):
                rng = Random(trial * 1000 + len(edge_set))
                c = find_long_cycle(rng, max_attempts=50, max_depth=30)
                if c is not None:
                    if best is None or len(c) > len(best):
                        best = c
            if best is None:
                # No cycle found; break to avoid infinite loop
                break
            pieces.append(best)
            remove_cycle(best)

        # Remaining edges as single pieces
        for u, v in edge_set:
            pieces.append([u, v])

        return pieces

    return [decompose(instance(*spec)) for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))]
