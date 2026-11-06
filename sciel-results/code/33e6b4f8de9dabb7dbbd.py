import time
import random
from collections import defaultdict
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_graph(n, edges, rng, deadline):
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        edge_set = {frozenset((u, v)) for u, v in edges}
        paths = []
        
        while edge_set:
            if time.time() > deadline:
                # If we run out of time, we must return a valid solution.
                # The current 'paths' list does not cover all edges.
                # We must append the remaining edges as individual paths to ensure validity.
                for e in edge_set:
                    paths.append([e[0], e[1]])
                break
                
            deg = defaultdict(int)
            for e in edge_set:
                u, v = tuple(e)
                deg[u] += 1
                deg[v] += 1
            
            start_candidates = [v for v in deg if deg[v] == 1]
            if not start_candidates:
                start_candidates = [v for v in deg if deg[v] > 0]
            
            if not start_candidates:
                break
                
            start_v = rng.choice(start_candidates)
            current = start_v
            path = [current]
            
            while True:
                neighbors = []
                for v in adj[current]:
                    e = frozenset((current, v))
                    if e in edge_set:
                        neighbors.append((v, deg[v]))
                
                if not neighbors:
                    break
                
                neighbors.sort(key=lambda x: x[1])
                min_deg = neighbors[0][1]
                candidates = [v for v, d in neighbors if d == min_deg]
                next_v = rng.choice(candidates)
                
                e = frozenset((current, next_v))
                edge_set.remove(e)
                path.append(next_v)
                
                deg[current] -= 1
                deg[next_v] -= 1
                current = next_v
                
            paths.append(path)
            
        return paths

    def solve_instance(n, p, seed, deadline):
        edges = instance(n, p, seed)
        if not edges:
            return []
        
        best_paths = None
        best_count = float('inf')
        
        # Dynamic number of restarts
        # Estimate time per restart
        # For now, just run as many as we can
        restarts = 0
        while time.time() < deadline - 2.0:
            restarts += 1
            rng = Random(seed + restarts)
            paths = decompose_graph(n, edges, rng, deadline)
            count = len(paths)
            if count < best_count:
                best_count = count
                best_paths = paths
                
        if best_paths is None:
            best_paths = [[list(e) for e in edges]]
            
        return best_paths

    results = []
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    
    for spec in specs:
        if time.time() > deadline:
            n, p, seed = spec
            edges = instance(n, p, seed)
            results.append([[list(e) for e in edges]])
            continue
            
        paths = solve_instance(*spec, deadline)
        results.append(paths)
        
    return results
