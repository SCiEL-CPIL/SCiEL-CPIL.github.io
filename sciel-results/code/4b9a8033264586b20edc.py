"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def construct():
    # Deadline guard: keep a safety margin before the 150s kill.
    T_START = time.time()
    DEADLINE = 130.0  # 20s safety margin

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def _decompose_graph(edges):
        """
        Decompose a graph into edge-disjoint cycles and single edges, minimizing piece count.
        Strategy:
          1. Build adjacency.
          2. Repeatedly find a cycle in the remaining graph using a DFS/heuristic that
             prefers longer cycles (to pack more edges per piece).
          3. Remove cycle edges and record the cycle as a piece.
          4. When no cycle remains, each remaining edge is its own piece.
        """
        # Build adjacency structure
        adj = defaultdict(set)
        edge_set = set()
        for (u, v) in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((min(u, v), max(u, v)))

        pieces = []
        remaining_edges = set(edge_set)

        # Helper to check if an edge still exists
        def has_edge(u, v):
            return (min(u, v), max(u, v)) in remaining_edges

        # Helper to remove an edge
        def remove_edge(u, v):
            key = (min(u, v), max(u, v))
            if key in remaining_edges:
                remaining_edges.discard(key)
                adj[u].discard(v)
                adj[v].discard(u)

        # Find a cycle starting from a given vertex using DFS
        def find_cycle_from(start):
            """
            DFS to find a cycle back to start.
            Returns list of vertices in the cycle (start, ..., start) or None.
            """
            # We want to find a path from start back to start using distinct vertices (except start).
            # Use DFS with backtracking, but limit depth to avoid explosion.
            # Prefer neighbors with higher degree to find longer cycles.
            
            # Sort neighbors by degree descending
            def neighbor_sort_key(v):
                return -len(adj[v])
            
            # DFS with visited set
            visited = set()
            path = [start]
            visited.add(start)
            
            def dfs(current, depth):
                if depth > 60:  # Limit depth to avoid long searches
                    return None
                for nxt in sorted(adj[current], key=neighbor_sort_key):
                    if nxt == start and len(path) >= 3:
                        # Found a cycle
                        return path[:]
                    if nxt not in visited:
                        visited.add(nxt)
                        path.append(nxt)
                        result = dfs(nxt, depth + 1)
                        if result is not None:
                            return result
                        path.pop()
                        visited.discard(nxt)
                return None
            
            return dfs(start, 0)

        # Main loop: extract cycles
        # We'll try to find cycles from high-degree vertices first
        while remaining_edges:
            if time.time() > DEADLINE:
                break
            
            # Find a vertex with maximum degree among remaining edges
            # Build degree map from remaining_edges
            deg = defaultdict(int)
            for (u, v) in remaining_edges:
                deg[u] += 1
                deg[v] += 1
            
            if not deg:
                break
            
            # Try each vertex in order of decreasing degree
            candidates = sorted(deg.keys(), key=lambda x: -deg[x])
            
            found_cycle = None
            for start in candidates:
                if deg[start] < 2:
                    break
                cycle = find_cycle_from(start)
                if cycle is not None:
                    found_cycle = cycle
                    break
            
            if found_cycle is None:
                # No more cycles; add remaining edges as single pieces
                for (u, v) in remaining_edges:
                    pieces.append([u, v])
                break
            else:
                # Extract the cycle
                # found_cycle is [v0, v1, ..., vk] where v0 == start, and we need to close the loop
                # Actually, find_cycle_from returns path where last step goes back to start
                # Let's re-examine: the DFS returns path when it finds nxt == start.
                # At that point, path is [start, ..., current] and current has edge to start.
                # So the cycle is path + [start] but we store it as the vertex list without repeating start.
                cycle_vertices = found_cycle[:]
                # Remove edges of the cycle
                for i in range(len(cycle_vertices)):
                    u = cycle_vertices[i]
                    v = cycle_vertices[(i + 1) % len(cycle_vertices)]
                    remove_edge(u, v)
                pieces.append(cycle_vertices)

        # If we broke out early due to deadline, add remaining edges as single pieces
        if remaining_edges:
            for (u, v) in remaining_edges:
                pieces.append([u, v])

        return pieces

    # Process the three fixed instances
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    for spec in specs:
        if time.time() > DEADLINE:
            # Fallback: just return all edges as single pieces for remaining instances
            edges = instance(*spec)
            results.append([[u, v] for (u, v) in edges])
            continue
        edges = instance(*spec)
        pieces = _decompose_graph(edges)
        results.append(pieces)

    return results

# EVOLVE-BLOCK-END
