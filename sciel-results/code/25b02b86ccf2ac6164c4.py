import time
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []

    for spec in specs:
        if time.time() > deadline:
            # Fallback: return empty lists or best effort? 
            # The task requires valid output. If we run out of time, we should return 
            # a valid decomposition for the ones we finished, and for the last one, 
            # maybe just return all edges as singletons quickly.
            pass
            
        edges = instance(*spec)
        n = spec[0]
        
        # Build adjacency list
        adj = defaultdict(set)
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v))

        pieces = []
        
        while edge_set:
            if time.time() > deadline:
                # Emergency fallback: add all remaining edges as singletons
                for e in edge_set:
                    pieces.append(list(e))
                break
                
            # Find a cycle
            cycle = None
            # We iterate through nodes to start DFS
            # To make it deterministic and fast, we can just pick the first edge in edge_set
            # and check if it's part of a cycle, or just run DFS from any node.
            
            # Let's use a DFS that finds any cycle in the remaining graph.
            # We need to track parent to avoid trivial backtracking.
            
            visited = [0] * n  # 0: unvisited, 1: visiting, 2: visited
            parent = [-1] * n
            
            def dfs(u):
                nonlocal cycle
                visited[u] = 1
                for v in list(adj[u]):
                    if (u, v) not in edge_set and (v, u) not in edge_set:
                        continue # Edge removed
                    
                    if visited[v] == 0:
                        parent[v] = u
                        if dfs(v):
                            return True
                    elif visited[v] == 1 and v != parent[u]:
                        # Found a cycle: v -> ... -> u
                        cycle_nodes = [u]
                        curr = u
                        while curr != v:
                            curr = parent[curr]
                            cycle_nodes.append(curr)
                        # cycle_nodes is [u, ..., v]
                        # The cycle is v -> ... -> u -> v
                        # We want the list of vertices in order.
                        # The path is v -> ... -> u. The edge u-v closes it.
                        # So the cycle vertices are [v, ..., u]
                        cycle = cycle_nodes[::-1] # [v, ..., u]
                        return True
                visited[u] = 2
                return False

            found_cycle = False
            for u in range(n):
                if visited[u] == 0:
                    if dfs(u):
                        found_cycle = True
                        break
            
            if found_cycle and cycle:
                # Extract cycle edges
                cycle_edges = set()
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    e = tuple(sorted((u, v)))
                    cycle_edges.add(e)
                
                # Remove from edge_set and adj
                for e in cycle_edges:
                    edge_set.remove(e)
                    u, v = e
                    adj[u].remove(v)
                    adj[v].remove(u)
                
                pieces.append(cycle)
            else:
                # No more cycles, add all remaining edges as singletons
                for e in edge_set:
                    pieces.append(list(e))
                break

        results.append(pieces)

    return results
