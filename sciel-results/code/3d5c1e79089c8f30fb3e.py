"""Decompose three fixed graphs into few cycles and edges."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    DEADLINE = 150.0 - 5.0  # 145s wall-clock safety margin

    # --- Instance generation (copied from verifier) ---
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))

    def time_left():
        return DEADLINE - (time.time() - start_time)

    def decompose_graph(edges):
        """
        Decompose a graph into cycles and single edges.
        Strategy:
          1. Build adjacency structure.
          2. Repeatedly find and remove cycles (prefer longer ones).
          3. Remaining edges become single-edge pieces.
        Uses a randomized restart heuristic with multiple trials to find
        a decomposition with few total pieces.
        """
        if not edges:
            return []

        n = max(max(u, v) for u, v in edges) + 1
        best_pieces = None
        best_count = len(edges)  # trivial upper bound: all edges

        # We'll try several randomized strategies
        for trial in range(200):
            if time_left() < 5.0:
                break

            # Build adjacency list
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)

            # Work on a copy of the edge set
            remaining = set(edges)
            pieces = []

            # Find cycles repeatedly
            # We use a randomized DFS-based cycle finder
            visited_global = set()

            for _ in range(500):  # max cycle extractions
                if not remaining:
                    break

                # Pick a random edge to start
                start_edge = random.choice(list(remaining))
                start = start_edge[0]

                # Try to find a cycle containing start
                # Use DFS with a stack
                # We want to find a simple cycle that uses only remaining edges
                cycle = _find_cycle_from(start, remaining, adj, n)
                if cycle:
                    # Remove cycle edges from remaining
                    for i in range(len(cycle)):
                        a, b = cycle[i], cycle[(i + 1) % len(cycle)]
                        e = (min(a, b), max(a, b))
                        remaining.discard(e)
                    pieces.append(cycle)
                else:
                    # No cycle found from this start; try a few more starts
                    found_any = False
                    for _ in range(10):
                        if not remaining:
                            break
                        s = random.choice(list(remaining))[0]
                        c = _find_cycle_from(s, remaining, adj, n)
                        if c:
                            for i in range(len(c)):
                                a, b = c[i], c[(i + 1) % len(c)]
                                e = (min(a, b), max(a, b))
                                remaining.discard(e)
                            pieces.append(c)
                            found_any = True
                            break
                    if not found_any:
                        break

            # Remaining edges become single-edge pieces
            for e in remaining:
                pieces.append(list(e))

            total = len(pieces)
            if total < best_count:
                best_count = total
                best_pieces = pieces

            if best_count <= _lower_bound(edges):
                break

        return best_pieces

    def _lower_bound(edges):
        """Simple lower bound: at least ceil(|E| / max_cycle_len) but we can't know max cycle.
        Use |E| / 3 as a rough bound since cycles have at least 3 edges.
        Actually a better bound: number of odd-degree vertices / 2 + something... 
        Let's just use |E|//3 as a very loose bound for early stopping."""
        return max(1, len(edges) // 3)

    def _find_cycle_from(start, remaining, adj, n):
        """
        Find a simple cycle starting from 'start' using only edges in 'remaining'.
        Returns list of vertices in cycle order, or None.
        Uses randomized DFS.
        """
        # Build local adjacency from remaining edges
        local_adj = defaultdict(set)
        for u, v in remaining:
            local_adj[u].add(v)
            local_adj[v].add(u)

        # DFS to find a cycle back to start
        # We want a cycle that includes 'start'
        # Approach: DFS from start, track path; if we reach start again, we have a cycle.
        # But we need to avoid revisiting nodes in the current path.

        # Randomize neighbor order for diversity
        for node in local_adj:
            random.shuffle(list(local_adj[node]))

        path = [start]
        on_path = {start}
        parent = {}

        def dfs(node):
            if time_left() < 2.0:
                return None
            neighbors = list(local_adj[node])
            random.shuffle(neighbors)
            for nb in neighbors:
                if nb == start and len(path) >= 3:
                    # Found a cycle
                    return list(path)
                if nb not in on_path:
                    on_path.add(nb)
                    path.append(nb)
                    parent[nb] = node
                    result = dfs(nb)
                    if result:
                        return result
                    path.pop()
                    on_path.discard(nb)
            return None

        result = dfs(start)
        return result

    results = []
    for spec in specs:
        if time_left() < 2.0:
            # Not enough time; use trivial decomposition
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue

        edges = instance(*spec)
        pieces = decompose_graph(edges)
        results.append(pieces)

    return results
# EVOLVE-BLOCK-END
