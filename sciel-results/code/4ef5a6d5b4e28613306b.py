import time
from random import Random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        if not edges:
            return []
        
        edge_set = set((min(u,v), max(u,v)) for u,v in edges)
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        def deg(v):
            return len(adj.get(v, set()))
        
        def remove_edge(u, v):
            adj[u].remove(v)
            adj[v].remove(u)
            if not adj[u]:
                del adj[u]
            if not adj[v]:
                del adj[v]
        
        def beam_search_path(start, beam_width=50, max_depth=30):
            """Beam search for longest simple path from start."""
            beam = [([start], {start})]
            best_path = [start]
            best_len = 1
            
            for _ in range(max_depth):
                if time.time() > deadline - 5.0:
                    break
                new_beam = []
                for path, visited in beam:
                    current = path[-1]
                    neighbors = [n for n in adj.get(current, set()) if n not in visited]
                    if not neighbors:
                        if len(path) > best_len:
                            best_len = len(path)
                            best_path = path[:]
                        continue
                    # Score neighbors by fewest unvisited neighbors (Warnsdorff)
                    neighbors.sort(key=lambda n: len([x for x in adj.get(n, set()) if x not in visited]))
                    # Take top few
                    for n in neighbors[:min(5, len(neighbors))]:
                        new_path = path + [n]
                        new_visited = visited | {n}
                        new_beam.append((new_path, new_visited))
                        if len(new_path) > best_len:
                            best_len = len(new_path)
                            best_path = new_path[:]
                
                if not new_beam:
                    break
                new_beam.sort(key=lambda x: -len(x[0]))
                beam = new_beam[:beam_width]
            
            return best_path
        
        def find_best_path():
            """Find best path by trying multiple starts."""
            if not adj:
                return None
            
            candidates = sorted(adj.keys(), key=lambda v: (deg(v), v))
            best = None
            best_len = 0
            
            num_starts = min(len(candidates), 12)
            for start in candidates[:num_starts]:
                if time.time() > deadline - 5.0:
                    break
                path = beam_search_path(start)
                if path and len(path) > best_len:
                    best_len = len(path)
                    best = path
            
            if best is None and adj:
                u = next(iter(adj))
                v = next(iter(adj[u]))
                best = [u, v]
            
            return best
        
        paths = []
        
        while adj:
            if time.time() > deadline - 3.0:
                while adj:
                    u = next(iter(adj))
                    v = next(iter(adj[u]))
                    paths.append([u, v])
                    remove_edge(u, v)
                break
            
            path = find_best_path()
            if path is None or len(path) < 2:
                u = next(iter(adj))
                v = next(iter(adj[u]))
                path = [u, v]
            
            paths.append(path)
            for i in range(len(path) - 1):
                remove_edge(path[i], path[i+1])
        
        # Efficient local search
        def build_used_edges():
            used = set()
            for p in paths:
                for i in range(len(p)-1):
                    used.add((min(p[i],p[i+1]), max(p[i],p[i+1])))
            return used
        
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 10.0:
                break
            used_edges = build_used_edges()
            for i in range(len(paths)):
                if time.time() > deadline - 10.0:
                    break
                for j in range(i+1, len(paths)):
                    if time.time() > deadline - 10.0:
                        break
                    p1, p2 = paths[i], paths[j]
                    if len(p1) < 2 or len(p2) < 2:
                        continue
                    # Try 4 endpoint combinations
                    for u, v, new_path in [
                        (p1[-1], p2[0], p1 + p2),
                        (p1[0], p2[-1], p2[::-1] + p1),
                        (p1[-1], p2[-1], p1 + p2[::-1]),
                        (p1[0], p2[0], p1[::-1] + p2),
                    ]:
                        e = (min(u,v), max(u,v))
                        if e in edge_set and e not in used_edges:
                            if len(set(new_path)) == len(new_path):
                                paths[i] = new_path
                                paths.pop(j)
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)
        
    return results
