import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin for 120s limit
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_longest_cycle(adj, n, remaining_edges):
        best_len = 0
        best_cycle = []
        
        for start in range(n):
            if not adj[start]:
                continue
            
            visited = {start: 0}
            parent = {start: None}
            in_stack = {start}
            
            # Iterative DFS to avoid recursion limits and allow early stopping if needed
            # But recursion is fine for N=30. Let's use recursion for clarity.
            # We need to pass state around.
            
            def dfs(u):
                nonlocal best_len, best_cycle
                for v in sorted(adj[u]):
                    if (min(u, v), max(u, v)) not in remaining_edges:
                        continue
                    if v not in visited:
                        visited[v] = visited[u] + 1
                        parent[v] = u
                        in_stack.add(v)
                        dfs(v)
                        in_stack.remove(v)
                    elif v in in_stack:
                        length = visited[u] - visited[v] + 1
                        if length >= 3 and length > best_len:
                            # Reconstruct
                            cycle = [u]
                            cur = u
                            while cur != v:
                                cur = parent[cur]
                                cycle.append(cur)
                            best_cycle = cycle
                            best_len = length

            dfs(start)
            
            if time.time() > deadline:
                break

        return best_cycle

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    for spec in specs:
        if time.time() > deadline:
            # Fallback: return empty or single edges?
            # If we run out of time, we should return what we have.
            # But we need to return 3 lists.
            # We'll just break and pad with empty lists or single edges for remaining.
            pass
            
        edges = instance(*spec)
        n = spec[0]
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        pieces = []
        remaining_edges = set(edges)
        
        while remaining_edges:
            if time.time() > deadline:
                # Time out: add remaining as single edges
                for e in remaining_edges:
                    pieces.append(list(e))
                break
                
            best_cycle = find_longest_cycle(adj, n, remaining_edges)
            if best_cycle:
                pieces.append(best_cycle)
                # Remove edges
                for i in range(len(best_cycle)):
                    u = best_cycle[i]
                    v = best_cycle[(i + 1) % len(best_cycle)]
                    e = (min(u, v), max(u, v))
                    if e in remaining_edges:
                        remaining_edges.discard(e)
                        adj[u].remove(v)
                        adj[v].remove(u)
            else:
                # No cycles, add singles
                for e in remaining_edges:
                    pieces.append(list(e))
                break

        results.append(pieces)
        
    return results
