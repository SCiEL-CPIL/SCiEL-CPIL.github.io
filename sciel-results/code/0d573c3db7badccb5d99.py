import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    for n, p, seed in specs:
        if time.time() > deadline:
            # Fallback: just return edges as single pieces for remaining graphs
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
            
        edges = instance(n, p, seed)
        
        # Build adjacency list
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        pieces = []
        
        def find_cycle():
            # Find a cycle in the current graph defined by adj
            # Returns a list of vertices [v1, v2, ..., vk] such that edges (v1,v2)...(vk,v1) exist
            # Or None
            
            # Start from highest degree node
            start = max(range(n), key=lambda x: len(adj[x]))
            if not adj[start]:
                return None
                
            visited = {start: -1} # node: parent
            parent = {start: None}
            path = [start]
            
            # We need to track the path to reconstruct the cycle
            # Standard DFS
            stack = [(start, iter(sorted(adj[start], key=lambda x: len(adj[x]), reverse=True)))]
            # Using a stack for iterative DFS to avoid recursion limits and to control order
            
            # Actually, recursive is easier for path tracking if depth is small (n=30).
            # Let's use recursion with sys.setrecursionlimit if needed, but n=30 is safe.
            
            def dfs(u):
                if time.time() > deadline:
                    return None
                for v in sorted(adj[u], key=lambda x: len(adj[x]), reverse=True):
                    if v == parent[u]:
                        continue
                    if v not in visited:
                        visited[v] = u
                        parent[v] = u
                        path.append(v)
                        cycle = dfs(v)
                        if cycle:
                            return cycle
                        path.pop()
                    else:
                        # Back edge found
                        # Cycle is path from v to u + edge (u,v)
                        # Find v in path
                        if v in path:
                            idx = path.index(v)
                            cycle_nodes = path[idx:]
                            # cycle_nodes is [v, ..., u]
                            # The cycle is v -> ... -> u -> v
                            return cycle_nodes
                return None
                
            cycle = dfs(start)
            return cycle

        while edges:
            if time.time() > deadline:
                # Add remaining as single edges
                for e in edges:
                    pieces.append(list(e))
                break
                
            cycle = find_cycle()
            if cycle:
                # Extract edges from cycle
                cycle_edges = set()
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    edge = (min(u, v), max(u, v))
                    cycle_edges.add(edge)
                
                # Remove edges
                for e in cycle_edges:
                    edges.remove(e)
                    u, v = e
                    adj[u].remove(v)
                    adj[v].remove(u)
                    
                pieces.append(cycle)
            else:
                # No more cycles
                for e in edges:
                    pieces.append(list(e))
                break
                
        results.append(pieces)
        
    return results
