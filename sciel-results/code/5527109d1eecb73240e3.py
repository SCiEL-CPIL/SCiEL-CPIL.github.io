import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = set()
        for i in range(n - 1):
            u, v = order[i], order[i + 1]
            edges.add((min(u, v), max(u, v)))
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return list(edges)

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    all_results = []

    for spec in specs:
        n, p, seed = spec
        edges = instance(n, p, seed)
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        deg = {i: 0 for i in range(n)}
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
            
        remaining_edges = set(edges)
        paths = []
        
        # 1. Handle odd degree vertices
        odd_vertices = [v for v in range(n) if deg[v] % 2 != 0]
        random.shuffle(odd_vertices)
        
        for i in range(0, len(odd_vertices), 2):
            if i + 1 >= len(odd_vertices):
                break
            u = odd_vertices[i]
            v = odd_vertices[i+1]
            
            path = find_simple_path_bfs(u, v, remaining_edges, adj)
            
            if path:
                paths.append(path)
                for j in range(len(path) - 1):
                    a, b = path[j], path[j+1]
                    e = (min(a, b), max(a, b))
                    remaining_edges.discard(e)
                    deg[a] -= 1
                    deg[b] -= 1
            else:
                for nb in list(adj[u]):
                    e = (min(u, nb), max(u, nb))
                    if e in remaining_edges:
                        paths.append([u, nb])
                        remaining_edges.discard(e)
                        deg[u] -= 1
                        deg[nb] -= 1
                        break

        # 2. Handle remaining Eulerian graph
        curr_edges = set(remaining_edges)
        
        while curr_edges:
            if time.time() > deadline:
                for e in curr_edges:
                    paths.append([e[0], e[1]])
                break
            
            active_nodes = set()
            for e in curr_edges:
                active_nodes.add(e[0])
                active_nodes.add(e[1])
            
            best_path = None
            starts = list(active_nodes)
            random.shuffle(starts)
            
            for s in starts[:10]:
                if time.time() > deadline:
                    break
                p = find_longest_simple_path_fast(s, curr_edges, adj)
                if p and (best_path is None or len(p) > len(best_path)):
                    best_path = p
            
            if best_path:
                paths.append(best_path)
                for j in range(len(best_path) - 1):
                    a, b = best_path[j], best_path[j+1]
                    e = (min(a, b), max(a, b))
                    curr_edges.discard(e)
            else:
                e = next(iter(curr_edges))
                paths.append([e[0], e[1]])
                curr_edges.discard(e)

        all_results.append(paths)

    return all_results

def find_simple_path_bfs(u, v, edges, adj):
    if u == v:
        return [u]
    visited = {u}
    queue = deque([(u, [u])])
    while queue:
        node, path = queue.popleft()
        if node == v:
            return path
        for nb in adj[node]:
            e = (min(node, nb), max(node, nb))
            if e in edges and nb not in visited:
                visited.add(nb)
                queue.append((nb, path + [nb]))
    return None

def find_longest_simple_path_fast(start, edges, adj):
    beam = [([start], {start}, start)]
    best = [start]
    
    for _ in range(len(adj)):
        new_beam = []
        for path, visited, node in beam:
            valid_neighbors = []
            for nb in adj[node]:
                if nb in visited:
                    continue
                e = (min(node, nb), max(node, nb))
                if e in edges:
                    count = 0
                    for m in adj[nb]:
                        if m in visited:
                            continue
                        e2 = (min(nb, m), max(nb, m))
                        if e2 in edges:
                            count += 1
                    valid_neighbors.append((count, nb))
            
            if not valid_neighbors:
                if len(path) > len(best):
                    best = path[:]
                continue
            
            valid_neighbors.sort(key=lambda x: x[0])
            for _, nb in valid_neighbors[:2]:
                new_path = path + [nb]
                new_visited = visited | {nb}
                new_beam.append((new_path, new_visited, nb))
                
        if not new_beam:
            break
        
        new_beam.sort(key=lambda x: -len(x[0]))
        beam = new_beam[:2]
        
        if beam and len(beam[0][0]) > len(best):
            best = beam[0][0][:]
            
    return best
