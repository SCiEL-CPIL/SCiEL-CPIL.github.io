import time
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s margin
    
    n = P['n']
    adj = defaultdict(list)
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    for u in adj:
        adj[u].sort()

    def find_ham(adj_map, forbidden_edges):
        start = 0
        path = [start]
        used_edges = set()
        
        def is_allowed(u, v):
            # Check if edge exists in adj_map (it does by construction)
            # Check if not forbidden
            if (u, v) in forbidden_edges or (v, u) in forbidden_edges:
                return False
            # Check if not used in current path
            if (u, v) in used_edges or (v, u) in used_edges:
                return False
            return True

        def dfs(current, path, used_edges):
            if len(path) == n:
                if is_allowed(current, start):
                    return path
                return None
            
            # Heuristic: Sort neighbors by number of unvisited neighbors
            # This is Warnsdorff's rule
            neighbors = []
            for v in adj_map[current]:
                if not is_allowed(current, v):
                    continue
                
                # If v is start, we can only go there at the end
                if v == start:
                    if len(path) == n - 1:
                        neighbors.append((0, v))
                    continue
                
                # Count unvisited neighbors of v
                unvisited = 0
                for w in adj_map[v]:
                    if w not in path and w != start:
                        if is_allowed(v, w):
                            unvisited += 1
                # If v has no unvisited neighbors (and isn't start), it's a dead end unless it's the last one
                # But we are not at the last step yet (len(path) < n-1)
                neighbors.append((unvisited, v))
            
            neighbors.sort(key=lambda x: x[0])
            
            for _, v in neighbors:
                edge = (min(current, v), max(current, v))
                used_edges.add(edge)
                path.append(v)
                
                result = dfs(v, path, used_edges)
                if result is not None:
                    return result
                
                path.pop()
                used_edges.remove(edge)
                
            return None
        
        return dfs(start, path, used_edges)

    # 1. Try deterministic
    cycles = []
    forbidden = set()
    for _ in range(3):
        c = find_ham(adj, forbidden)
        if c is None:
            break
        cycles.append(c)
        for i in range(len(c)):
            u = c[i]
            v = c[(i + 1) % len(c)]
            forbidden.add((min(u, v), max(u, v)))
            
    if len(cycles) == 3:
        return cycles

    # 2. Try randomized
    # We will shuffle the adjacency lists for each attempt
    # To avoid re-creating the function, we pass adj_map
    
    for attempt in range(50):
        if time.time() > deadline:
            break
            
        # Create a copy of adj and shuffle
        adj_copy = {u: adj[u][:] for u in adj}
        for u in adj_copy:
            random.shuffle(adj_copy[u])
            
        cycles = []
        forbidden = set()
        for _ in range(3):
            c = find_ham(adj_copy, forbidden)
            if c is None:
                break
            cycles.append(c)
            for i in range(len(c)):
                u = c[i]
                v = c[(i + 1) % len(c)]
                forbidden.add((min(u, v), max(u, v)))
                
        if len(cycles) == 3:
            return cycles

    # If we found fewer than 3, return what we have
    # Ensure we return at least 1 if possible
    if not cycles:
        c = find_ham(adj, set())
        if c:
            return [c]
            
    return cycles
