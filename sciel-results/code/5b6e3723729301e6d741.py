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

    def try_merge_into_cycles(pieces, n, adj, remaining_edges):
        """Try to merge single edges into cycles by finding paths between their endpoints."""
        improved = False
        i = 0
        while i < len(pieces) and time.time() < deadline:
            if len(pieces[i]) == 2:
                u, v = pieces[i]
                if (min(u, v), max(u, v)) in remaining_edges:
                    # Try to find a path from u to v using remaining edges (excluding direct edge)
                    path = find_path(u, v, adj, remaining_edges, exclude_edge=(min(u, v), max(u, v)))
                    if path and len(path) >= 3:
                        # Replace single edge with cycle
                        pieces[i] = path
                        for j in range(len(path)):
                            a, b = path[j], path[(j + 1) % len(path)]
                            e = (min(a, b), max(a, b))
                            if e in remaining_edges:
                                remaining_edges.discard(e)
                                adj[a].remove(b)
                                adj[b].remove(a)
                        improved = True
                    else:
                        i += 1
                else:
                    i += 1
            else:
                i += 1
        return improved

    def find_path(start, end, adj, remaining_edges, exclude_edge=None):
        """Find a path from start to end using remaining edges."""
        from collections import deque
        if start == end:
            return None
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue and time.time() < deadline:
            node, path = queue.popleft()
            for neighbor in sorted(adj[node]):
                e = (min(node, neighbor), max(node, neighbor))
                if e not in remaining_edges:
                    continue
                if e == exclude_edge:
                    continue
                if neighbor == end:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

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
                for e in remaining_edges:
                    pieces.append(list(e))
                break
                
            best_cycle = find_longest_cycle(adj, n, remaining_edges)
            if best_cycle:
                pieces.append(best_cycle)
                for i in range(len(best_cycle)):
                    u = best_cycle[i]
                    v = best_cycle[(i + 1) % len(best_cycle)]
                    e = (min(u, v), max(u, v))
                    if e in remaining_edges:
                        remaining_edges.discard(e)
                        adj[u].remove(v)
                        adj[v].remove(u)
            else:
                for e in remaining_edges:
                    pieces.append(list(e))
                break

        # Post-processing: try to merge single edges into cycles
        if time.time() < deadline:
            try_merge_into_cycles(pieces, n, adj, remaining_edges)

        results.append(pieces)
        
    return results
