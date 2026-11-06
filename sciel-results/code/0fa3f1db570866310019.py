import time
from collections import defaultdict
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def get_degrees(edges):
        degs = defaultdict(int)
        for u, v in edges:
            degs[u] += 1
            degs[v] += 1
        return degs

    def eulerian_path(edges, start=None):
        if not edges:
            return []
        
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Find start
        if start is None:
            odd_vertices = [v for v, d in get_degrees(edges).items() if d % 2 != 0]
            if odd_vertices:
                start = odd_vertices[0]
            else:
                start = next(iter(adj))
        
        path = [start]
        stack = [start]
        remaining = set(edges)
        
        # Hierholzer's algorithm
        while stack:
            v = stack[-1]
            # Find an unused edge
            next_v = None
            while adj[v]:
                u = adj[v].pop()
                e = tuple(sorted((v, u)))
                if e in remaining:
                    remaining.remove(e)
                    # Remove reverse edge
                    adj[u].remove(v)
                    next_v = u
                    break
            if next_v is not None:
                stack.append(next_v)
                path.append(next_v)
            else:
                stack.pop()
                
        return path

    def decompose_graph(edges):
        if not edges:
            return []
            
        degs = get_degrees(edges)
        odd_vertices = [v for v, d in degs.items() if d % 2 != 0]
        K = max(1, len(odd_vertices) // 2)
        
        # If K=1, we can just use Eulerian path
        if K == 1:
            return [eulerian_path(edges)]
            
        best_paths = None
        best_count = float('inf')
        
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        original_edges = set(edges)
        
        # Randomized Greedy with Restart
        rng = Random(42)
        
        for attempt in range(200):
            if time.time() > deadline:
                break
                
            current_adj = defaultdict(list)
            for u, v in edges:
                current_adj[u].append(v)
                current_adj[v].append(u)
                
            remaining_edges = set(edges)
            paths = []
            
            while remaining_edges:
                # Pick a start vertex
                # Prefer vertices with high degree to start long paths?
                # Or low degree to clear leaves?
                # Let's pick a random vertex from those with remaining edges.
                active_vertices = set()
                for u, v in remaining_edges:
                    active_vertices.add(u)
                    active_vertices.add(v)
                
                if not active_vertices:
                    break
                    
                start = rng.choice(list(active_vertices))
                path = [start]
                current = start
                visited = {start}
                
                while True:
                    # Find neighbors with remaining edges
                    avail = [nb for nb in current_adj[current] 
                             if tuple(sorted((current, nb))) in remaining_edges 
                             and nb not in visited]
                    
                    if not avail:
                        break
                    
                    # Heuristic: Pick neighbor with highest remaining degree
                    # This tends to keep the graph connected and allows longer paths
                    avail.sort(key=lambda x: len([e for e in remaining_edges if x in e]), reverse=True)
                    
                    # Add some randomness to break ties
                    if len(avail) > 1 and rng.random() < 0.2:
                        next_v = rng.choice(avail[:3])
                    else:
                        next_v = avail[0]
                    
                    edge = tuple(sorted((current, next_v)))
                    remaining_edges.remove(edge)
                    
                    # Update adjacency
                    current_adj[current].remove(next_v)
                    current_adj[next_v].remove(current)
                    
                    path.append(next_v)
                    visited.add(next_v)
                    current = next_v
                    
                paths.append(path)
                
            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = paths
                
                if best_count == K:
                    break
                    
        if best_paths is None:
            # Fallback: individual edges
            return [list(e) for e in edges]
            
        return best_paths

    results = []
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    
    for spec in specs:
        edges = instance(*spec)
        paths = decompose_graph(edges)
        results.append(paths)
        
    return results
