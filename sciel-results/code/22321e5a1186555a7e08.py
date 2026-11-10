"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = P['n']
    edges = P['edges']
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    def find_max_clique():
        best = []
        def bk(R, P, X):
            if not P and not X:
                if len(R) > len(best):
                    best[:] = R[:]
                return
            if P:
                u = P[0]
                candidates = [v for v in P if v not in adj[u]]
            else:
                candidates = P[:]
            for v in candidates:
                bk(R + [v], [x for x in P if x in adj[v]], [x for x in X if x in adj[v]])
                P.remove(v)
                X.append(v)
        bk([], list(range(n)), [])
        return best

    initial_clique = find_max_clique()
    
    def find_path(u, v, forbidden):
        if u == v:
            return [u, v]
        visited = {u}
        queue = deque([(u, [u])])
        while queue:
            curr, path = queue.popleft()
            for neighbor in adj[curr]:
                if neighbor == v:
                    return path + [neighbor]
                if neighbor not in visited and neighbor not in forbidden:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    def get_interiors(paths_dict):
        inters = set()
        for p in paths_dict.values():
            inters.update(p[1:-1])
        return inters

    def try_add_vertex(v, B, paths_dict):
        # Try to add v to B
        # We need paths from v to each b in B
        # These paths must be disjoint from each other and from existing paths
        existing_interiors = get_interiors(paths_dict)
        
        # Try multiple random orders of connecting to B
        for _ in range(10): # 10 attempts
            targets = list(B)
            random.shuffle(targets)
            current_forbidden = set(existing_interiors) | set(B) | {v}
            new_paths = {}
            success = True
            
            for b in targets:
                p = find_path(v, b, current_forbidden)
                if p is None:
                    success = False
                    break
                new_paths[b] = p
                for x in p[1:-1]:
                    current_forbidden.add(x)
                    
            if success:
                # Merge into paths_dict
                new_paths_dict = dict(paths_dict)
                for b in B:
                    key = tuple(sorted((v, b)))
                    new_paths_dict[key] = new_paths[b]
                return B + [v], new_paths_dict
                
        return None

    best_B = initial_clique
    best_paths = {}
    for u, v in combinations(initial_clique, 2):
        key = tuple(sorted((u, v)))
        best_paths[key] = [u, v]
        
    best_size = len(best_B)
    
    # Expansion loop
    while time.time() < deadline:
        expanded = False
        # Try to add any vertex not in best_B
        for v in range(n):
            if v in best_B:
                continue
            result = try_add_vertex(v, best_B, best_paths)
            if result:
                best_B, best_paths = result
                best_size = len(best_B)
                expanded = True
                break # Restart expansion loop
                
        if not expanded:
            # Try randomized restarts with different initial sets?
            # Or just break if we can't expand
            break

    # Final check and format
    branches = [int(b) for b in best_B]
    paths = []
    for u, v in combinations(branches, 2):
        key = tuple(sorted((u, v)))
        paths.append(list(best_paths[key]))
        
    return {'branches': branches, 'paths': paths}

# EVOLVE-BLOCK-END
