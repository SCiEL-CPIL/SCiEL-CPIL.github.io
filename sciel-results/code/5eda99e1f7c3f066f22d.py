import time
from random import Random

def construct():
    start_time = time.time()
    DEADLINE = 140.0
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def build_adj(edge_set):
        adj = {}
        for u, v in edge_set:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        return adj
    
    def greedy_cycle(adj, start, max_depth=20):
        """Greedy: extend path from start, try to close back to start."""
        path = [start]
        on_path = {start}
        
        for _ in range(max_depth):
            # Find best next vertex
            best_next = None
            best_score = -1
            for neighbor in adj.get(path[-1], []):
                if neighbor in on_path:
                    if neighbor == start and len(path) >= 3:
                        return list(path)
                    continue
                score = len(adj.get(neighbor, []))
                if score > best_score:
                    best_score = score
                    best_next = neighbor
            
            if best_next is None:
                break
            path.append(best_next)
            on_path.add(best_next)
        
        # Check if we can close
        if len(path) >= 3 and start in adj.get(path[-1], []):
            return list(path)
        return None
    
    def find_any_cycle(adj):
        visited = set()
        
        def dfs(node, parent, path):
            visited.add(node)
            for neighbor in adj.get(node, []):
                if neighbor == parent:
                    continue
                if neighbor in visited:
                    idx = path.index(neighbor)
                    return path[idx:]
                path.append(neighbor)
                result = dfs(neighbor, node, path)
                if result:
                    return result
                path.pop()
            return None
        
        for v in sorted(adj.keys()):
            if v not in visited and adj[v]:
                result = dfs(v, -1, [v])
                if result:
                    return result
        return None
    
    def decompose(edges):
        if not edges:
            return []
        
        pieces = []
        remaining = set()
        for e in edges:
            remaining.add(tuple(sorted(e)))
        
        for _ in range(5000):
            if time.time() - start_time > DEADLINE:
                break
            if not remaining:
                break
            
            adj = build_adj(remaining)
            
            # Find vertex with max degree
            candidates = [v for v in adj if len(adj[v]) >= 2]
            if not candidates:
                break
            
            max_deg_vertex = max(candidates, key=lambda v: len(adj[v]))
            
            # Try greedy cycle from this vertex
            cycle = greedy_cycle(adj, max_deg_vertex)
            
            if cycle is None:
                # Try other high-degree vertices
                for v in sorted(candidates, key=lambda x: -len(adj[x])):
                    if time.time() - start_time > DEADLINE:
                        break
                    cycle = greedy_cycle(adj, v)
                    if cycle is not None:
                        break
            
            if cycle is None:
                # Fallback: find any cycle
                cycle = find_any_cycle(adj)
            
            if cycle is None:
                break
            
            # Remove cycle edges
            cycle_edges = set()
            valid = True
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i+1) % len(cycle)]
                key = tuple(sorted((u, v)))
                if key not in remaining:
                    valid = False
                    break
                cycle_edges.add(key)
            
            if valid:
                pieces.append(cycle)
                remaining -= cycle_edges
            else:
                break
        
        for e in remaining:
            pieces.append([e[0], e[1]])
        
        return pieces
    
    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    for spec in specs:
        if time.time() - start_time > DEADLINE:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        
        edges = instance(*spec)
        pieces = decompose(edges)
        results.append(pieces)
    
    return results
