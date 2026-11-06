"""Decompose graphs into minimum number of cycles and single edges."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    from collections import defaultdict, deque

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        # Build adjacency with edge indices for tracking
        # We'll work with a set of edges and repeatedly extract cycles
        remaining = set(edges)
        pieces = []
        
        # Build adjacency list from remaining edges
        def build_adj(rem):
            adj = defaultdict(list)
            for u, v in rem:
                adj[u].append(v)
                adj[v].append(u)
            return adj

        # Find a cycle in the graph using DFS
        def find_cycle(rem):
            adj = build_adj(rem)
            visited = set()
            parent = {}
            
            def dfs(start):
                stack = [(start, -1)]
                visited.add(start)
                parent[start] = -1
                while stack:
                    node, par = stack.pop()
                    for nb in adj[node]:
                        if nb == par:
                            continue
                        if nb in visited:
                            # Found a back edge, reconstruct cycle
                            cycle = [node]
                            cur = nb
                            while cur != node:
                                cur = parent[cur]
                                cycle.append(cur)
                            cycle.append(node)
                            return cycle
                        visited.add(nb)
                        parent[nb] = node
                        stack.append((nb, node))
                return None
            
            for start in range(max(rem[0]) + 1 if rem else 0):
                if start in visited:
                    continue
                # Only start from nodes that have edges
                if start not in adj:
                    continue
                result = dfs(start)
                if result:
                    return result
            return None

        # More robust cycle finding using iterative DFS with proper parent tracking
        def find_cycle_v2(rem):
            if not rem:
                return None
            adj = defaultdict(list)
            for u, v in rem:
                adj[u].append(v)
                adj[v].append(u)
            
            visited = set()
            
            for start in sorted(adj.keys()):
                if start in visited:
                    continue
                # DFS
                stack = [(start, None, [start])]
                visited.add(start)
                while stack:
                    node, par, path = stack.pop()
                    for nb in adj[node]:
                        if nb == par:
                            continue
                        if nb in visited:
                            # Cycle found: from nb to node in path
                            if nb in path:
                                idx = path.index(nb)
                                cycle = path[idx:]
                                return cycle
                        else:
                            visited.add(nb)
                            stack.append((nb, node, path + [nb]))
            return None

        # Even better: use the fact that we can find cycles by looking for 
        # any node with degree >= 2 and following edges
        def find_cycle_v3(rem):
            if not rem:
                return None
            adj = defaultdict(set)
            for u, v in rem:
                adj[u].add(v)
                adj[v].add(u)
            
            # Find any node with degree >= 2
            candidates = [n for n in adj if len(adj[n]) >= 2]
            if not candidates:
                return None
            
            for start in candidates:
                # Try to find a cycle starting from start
                # Use DFS with path tracking
                visited = set()
                stack = [(start, None, [start])]
                visited.add(start)
                while stack:
                    node, par, path = stack.pop()
                    for nb in sorted(adj[node], reverse=True):
                        if nb == par:
                            continue
                        if nb in visited:
                            if nb in path:
                                idx = path.index(nb)
                                cycle = path[idx:]
                                if len(cycle) >= 3:
                                    return cycle
                        else:
                            visited.add(nb)
                            stack.append((nb, node, path + [nb]))
            return None

        # Remove edges of a cycle from remaining
        def remove_cycle_edges(rem, cycle):
            new_rem = set(rem)
            for i in range(len(cycle) - 1):
                u, v = cycle[i], cycle[i + 1]
                edge = (u, v) if u < v else (v, u)
                new_rem.discard(edge)
            return new_rem

        # Greedy: repeatedly find and extract cycles
        # Prioritize longer cycles
        max_iter = 1000
        iteration = 0
        while remaining and iteration < max_iter:
            iteration += 1
            # Try to find the longest cycle
            best_cycle = None
            # Try multiple times with different starts to find longer cycles
            adj = defaultdict(set)
            for u, v in remaining:
                adj[u].add(v)
                adj[v].add(u)
            
            # Get nodes sorted by degree descending
            nodes_by_degree = sorted(adj.keys(), key=lambda x: -len(adj[x]))
            
            for start in nodes_by_degree[:20]:  # Try top 20 nodes
                visited = set()
                stack = [(start, None, [start])]
                visited.add(start)
                while stack:
                    node, par, path = stack.pop()
                    # Sort neighbors by degree for better cycles
                    for nb in sorted(adj[node], key=lambda x: -len(adj[x])):
                        if nb == par:
                            continue
                        if nb in visited:
                            if nb in path:
                                idx = path.index(nb)
                                cycle = path[idx:]
                                if len(cycle) >= 3:
                                    if best_cycle is None or len(cycle) > len(best_cycle):
                                        best_cycle = cycle
                        else:
                            visited.add(nb)
                            stack.append((nb, node, path + [nb]))
                if best_cycle and len(best_cycle) >= 5:
                    break
            
            if best_cycle and len(best_cycle) >= 3:
                pieces.append(best_cycle)
                remaining = remove_cycle_edges(remaining, best_cycle)
            else:
                break

        # Remaining edges are single edges
        for u, v in remaining:
            pieces.append([u, v])
        
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        pieces = decompose(edges)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
