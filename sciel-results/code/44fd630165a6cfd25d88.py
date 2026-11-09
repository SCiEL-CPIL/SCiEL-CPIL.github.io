"""Decompose fixed Erdős-Gallai instance graphs into few cycles and edges."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import sys

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(n, edges):
        # Build adjacency
        adj = {i: set() for i in range(n)}
        edge_set = set()
        for (u, v) in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v))

        pieces = []
        remaining = set(edge_set)

        def neighbors_of(v):
            # neighbors connected by a remaining edge
            return [w for w in adj[v] if (v, w) in remaining or (w, v) in remaining]

        def has_edge(a, b):
            return (a, b) in remaining or (b, a) in remaining

        def remove_edge(a, b):
            e = (a, b) if a < b else (b, a)
            remaining.discard(e)

        # Greedy long-cycle finder via DFS with backtracking, bounded
        def find_cycle(start, max_nodes_to_visit=20000):
            # DFS to find a cycle through start; try to make it long.
            # Returns list of vertices forming a cycle, or None.
            best = None
            # Iterative DFS with parent tracking to detect back-edge to start.
            # We'll do a depth-limited search trying to extend paths.
            stack = [(start, [start], {start})]
            visited_count = 0
            max_depth = 0
            while stack:
                node, path, onpath = stack.pop()
                visited_count += 1
                if visited_count > max_nodes_to_visit:
                    break
                if len(path) > max_depth:
                    max_depth = len(path)
                # Try to find a neighbor that is start (closing a cycle)
                # Prefer closing with a longer path first? We'll just close on first found
                # but also continue to try to extend for longer cycles.
                nbrs = [w for w in adj[node] if has_edge(node, w)]
                # Sort neighbors: prefer those not on path (to extend), but also check start
                # We want a cycle, so if start is a neighbor and len(path)>=3, we have a cycle.
                # To get a longer cycle, we might prefer to extend first.
                # Strategy: if we can extend (neighbor not on path), do that; else close.
                extendable = [w for w in nbrs if w not in onpath]
                can_close = (start in nbrs) and (len(path) >= 3)

                if extendable:
                    # Try extending; order by degree (prefer high-degree to find long cycles)
                    extendable.sort(key=lambda w: len(adj[w]), reverse=True)
                    for w in extendable:
                        new_path = path + [w]
                        new_onpath = onpath | {w}
                        stack.append((w, new_path, new_onpath))
                elif can_close:
                    # Found a cycle
                    cyc = path[:]  # path is start..node, closing edge node-start
                    if best is None or len(cyc) > len(best):
                        best = cyc
                    # Don't push more; but we might find longer by other branches already in stack
                # If neither, dead end, do nothing
            return best

        # Main loop: keep extracting cycles
        # To avoid infinite loop, limit total attempts
        max_attempts = 500
        attempts = 0
        while remaining and attempts < max_attempts:
            attempts += 1
            # Pick a starting vertex with highest degree among remaining
            candidates = [v for v in range(n) if adj[v]]
            # Use degree in remaining graph
            def rem_degree(v):
                return sum(1 for w in adj[v] if has_edge(v, w))

            candidates.sort(key=rem_degree, reverse=True)
            found = False
            for start in candidates:
                if rem_degree(start) < 2:
                    continue
                cyc = find_cycle(start, max_nodes_to_visit=30000)
                if cyc is not None and len(cyc) >= 3:
                    # Remove cycle edges
                    for i in range(len(cyc)):
                        a = cyc[i]
                        b = cyc[(i + 1) % len(cyc)]
                        remove_edge(a, b)
                    pieces.append(cyc)
                    found = True
                    break
            if not found:
                break

        # Remaining edges as single-edge pieces
        for (u, v) in sorted(remaining):
            pieces.append([u, v])

        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose(n, edges)
        results.append(pieces)
    return results
# EVOLVE-BLOCK-END
