import time
import random

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_graph(edges, n, max_restarts=100):
        if not edges:
            return []
        
        best_paths = None
        best_count = float('inf')
        
        # Use a fixed seed for the heuristic to be deterministic
        heuristic_rng = random.Random(42)
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            remaining_edges = set(edges)
            paths = []
            
            def get_neighbors(v):
                neighbors = []
                for u, w in remaining_edges:
                    if u == v:
                        neighbors.append(w)
                    elif w == v:
                        neighbors.append(u)
                return neighbors

            vertices_with_edges = set()
            for u, v in remaining_edges:
                vertices_with_edges.add(u)
                vertices_with_edges.add(v)
                
            if not vertices_with_edges:
                break
                
            start_candidates = list(vertices_with_edges)
            heuristic_rng.shuffle(start_candidates)
            
            current = start_candidates[0]
            path = [current]
            
            while True:
                neighbors = get_neighbors(current)
                if not neighbors:
                    break
                
                best_next = None
                max_deg = -1
                # Shuffle neighbors to break ties randomly
                heuristic_rng.shuffle(neighbors)
                
                for nb in neighbors:
                    deg = len(get_neighbors(nb))
                    if deg > max_deg:
                        max_deg = deg
                        best_next = nb
                        
                if best_next is None:
                    break
                
                e = (current, best_next)
                if e not in remaining_edges:
                    e = (best_next, current)
                remaining_edges.remove(e)
                
                path.append(best_next)
                current = best_next
                
            paths.append(path)
            
            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = [list(p) for p in paths]
                
            if best_count == (n - 1) // 2:
                break

        if best_paths is None:
            return [[list(e)] for e in edges]
            
        return best_paths

    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    results = []
    
    for n, p, seed in specs:
        if time.time() > deadline:
            break
            
        edges = instance(n, p, seed)
        paths = decompose_graph(edges, n, max_restarts=100)
        results.append(paths)
        
    while len(results) < len(specs):
        n, p, seed = specs[len(results)]
        edges = instance(n, p, seed)
        results.append([[list(e)] for e in edges])

    return results
