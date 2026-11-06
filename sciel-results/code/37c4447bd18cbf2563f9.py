"""Decompose random graphs into few cycles and edges."""

def construct():
    import random
    from collections import defaultdict

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        """
        Decompose an undirected graph into cycles and single edges.
        Strategy:
        1. Build adjacency list.
        2. While edges remain:
           a. Try to find a cycle in the current graph.
           b. If a cycle is found, remove its edges and record the cycle.
           c. If no cycle is found (graph is a forest), take any edge as a single-edge piece.
        
        To find cycles efficiently:
        - Use DFS to find a back edge, which indicates a cycle.
        - Extract the cycle from the DFS tree.
        
        However, a simpler heuristic that often works well for sparse random graphs:
        - Find a vertex with degree >= 2.
        - Follow a path until you revisit a vertex (forming a cycle).
        - Remove the cycle edges.
        - Repeat until no cycles remain, then take remaining edges as singles.
        
        Actually, the most robust simple approach:
        - While there are edges:
            - Find any cycle in the graph using a stack/DFS.
            - If found, output the cycle and remove its edges.
            - If not found, pick any edge, output it, and remove it.
        
        For finding a cycle:
        - Start from any vertex with degree > 0.
        - Do a DFS, keeping track of the parent and the current path.
        - When we encounter a back edge (to an ancestor in the DFS tree), we have a cycle.
        - The cycle is the portion of the path from the ancestor to the current vertex, plus the back edge.
        
        Let's implement this.
        """
        # Build adjacency list
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        pieces = []
        remaining_edges = set(edges)
        
        while remaining_edges:
            # Find a cycle
            cycle = None
            # Try to find a cycle starting from each vertex
            for start in list(adj.keys()):
                if not adj[start]:
                    continue
                # DFS to find a cycle
                visited = set()
                parent = {}
                path = []
                found = False
                cycle_vertices = []
                
                def dfs(u):
                    nonlocal found, cycle_vertices
                    if found:
                        return
                    visited.add(u)
                    path.append(u)
                    for v in sorted(adj[u]):  # sorted for determinism
                        if v == parent.get(u):
                            continue
                        if v in visited:
                            # Back edge found, extract cycle
                            found = True
                            # Find v in path
                            idx = path.index(v)
                            cycle_vertices = path[idx:]
                            return
                        if adj[v]:  # Only recurse if v has edges (optional optimization)
                            parent[v] = u
                            dfs(v)
                            if found:
                                return
                            path.pop()
                
                dfs(start)
                if found and cycle_vertices:
                    # cycle_vertices is a list of vertices forming a cycle
                    # The cycle is cycle_vertices[0] -> cycle_vertices[1] -> ... -> cycle_vertices[-1] -> cycle_vertices[0]
                    # We need to verify all edges exist in remaining_edges
                    valid = True
                    for i in range(len(cycle_vertices)):
                        u = cycle_vertices[i]
                        v = cycle_vertices[(i + 1) % len(cycle_vertices)]
                        edge = tuple(sorted((u, v)))
                        if edge not in remaining_edges:
                            valid = False
                            break
                    if valid and len(cycle_vertices) >= 3:
                        cycle = cycle_vertices
                        break
                    elif valid and len(cycle_vertices) == 2:
                        # This is a 2-cycle (multi-edge), but our graph is simple, so this shouldn't happen
                        # Unless we have parallel edges, which we don't in simple graphs
                        pass
                    # If invalid, continue searching from other starts
                # If no cycle found from this start, continue to next start
            
            if cycle is not None:
                # Remove the cycle edges
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    edge = tuple(sorted((u, v)))
                    remaining_edges.discard(edge)
                    adj[u].discard(v)
                    adj[v].discard(u)
                pieces.append(cycle)
            else:
                # No cycle found, take a single edge
                # Find any remaining edge
                edge = next(iter(remaining_edges))
                remaining_edges.discard(edge)
                u, v = edge
                adj[u].discard(v)
                adj[v].discard(u)
                pieces.append(list(edge))
        
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        pieces = decompose(edges)
        results.append(pieces)
    
    return results
