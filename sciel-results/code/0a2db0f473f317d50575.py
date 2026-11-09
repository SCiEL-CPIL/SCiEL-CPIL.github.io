import time
from random import Random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def solve_graph(edges):
        if not edges:
            return []
        
        n = max(max(u, v) for u, v in edges) + 1
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        remaining_edges = set(edges)
        paths = []
        
        def remaining_degree(v):
            deg = 0
            for u in adj[v]:
                e = (min(v, u), max(v, u))
                if e in remaining_edges:
                    deg += 1
            return deg

        while remaining_edges:
            if time.time() > deadline:
                # If we run out of time, we cannot return an incomplete solution.
                # We must ensure the current state is valid or finish quickly.
                # Given the speed of greedy, this is unlikely to trigger for n<=30.
                pass

            # Find start vertex
            candidates = [v for v in range(n) if remaining_degree(v) > 0]
            if not candidates:
                break
                
            odd_candidates = [v for v in candidates if remaining_degree(v) % 2 == 1]
            if odd_candidates:
                start = max(odd_candidates, key=lambda x: remaining_degree(x))
            else:
                start = max(candidates, key=lambda x: remaining_degree(x))
                
            path = [start]
            visited = {start}
            current = start
            
            while True:
                neighbors = []
                for u in adj[current]:
                    e = (min(current, u), max(current, u))
                    if e in remaining_edges and u not in visited:
                        neighbors.append(u)
                        
                if not neighbors:
                    break
                    
                next_v = max(neighbors, key=lambda x: remaining_degree(x))
                
                path.append(next_v)
                visited.add(next_v)
                
                e = (min(current, next_v), max(current, next_v))
                remaining_edges.remove(e)
                
                current = next_v
                
            paths.append(path)
            
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = solve_graph(edges)
        results.append(paths)
        
    return results
