import time
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    edge_set = set()
    for u, v in edges:
        edge_set.add((u, v))
        edge_set.add((v, u))

    def find_paths_limited(start, end, forbidden, max_len=6, max_paths=10):
        paths = []
        visited = {start}
        queue = deque([(start, [start])])
        while queue and len(paths) < max_paths:
            curr, path = queue.popleft()
            if len(path) > max_len:
                continue
            if curr == end:
                paths.append(path)
                continue
            for nei in adj[curr]:
                if nei not in visited and nei not in forbidden:
                    visited.add(nei)
                    queue.append((nei, path + [nei]))
        return paths

    def check_subdivision(branches, shuffle_pairs=True):
        k = len(branches)
        if k < 2:
            return None
        
        pairs = list(combinations(branches, 2))
        if shuffle_pairs:
            random.shuffle(pairs)
            
        paths = {}
        used_interiors = set()
        
        def backtrack(idx):
            if idx == len(pairs):
                return True
            
            u, v = pairs[idx]
            forbidden = used_interiors.union(set(branches) - {u, v})
            
            candidates = find_paths_limited(u, v, forbidden)
            # Sort by length
            candidates.sort(key=len)
            
            for path in candidates:
                interior = set(path[1:-1])
                if interior.isdisjoint(used_interiors):
                    paths[(u, v)] = path
                    used_interiors.update(interior)
                    if backtrack(idx + 1):
                        return True
                    paths.pop((u, v))
                    used_interiors.difference_update(interior)
            
            return False

        if backtrack(0):
            return paths
        return None

    def bron_kerbosch(R, P, X):
        if not P and not X:
            return R
        max_clique = []
        if P:
            pivot = max(P, key=lambda x: len(adj[x]))
            candidates = [v for v in P if (v, pivot) not in edge_set]
        else:
            candidates = P
            
        for v in candidates:
            new_R = R + [v]
            new_P = [x for x in P if (x, v) in edge_set]
            new_X = [x for x in X if (x, v) in edge_set]
            clique = bron_kerbosch(new_R, new_P, new_X)
            if len(clique) > len(max_clique):
                max_clique = clique
            P.remove(v)
            X.append(v)
        return max_clique

    max_clique = bron_kerbosch([], list(range(n)), [])
    best_size = len(max_clique)
    best_result = {'branches': max_clique, 'paths': []}
    
    paths = check_subdivision(max_clique)
    if paths:
        best_result['paths'] = [list(paths[k]) for k in paths]
    else:
        # Should not happen for a clique, but fallback
        best_result = {'branches': max_clique[:2], 'paths': [[max_clique[0], max_clique[1]]]}
        best_size = 2

    current_branches = list(max_clique)
    for v in range(n):
        if v in current_branches:
            continue
        new_branches = current_branches + [v]
        # Try a few times with shuffling
        for _ in range(5):
            paths = check_subdivision(new_branches)
            if paths:
                current_branches = new_branches
                best_size = len(current_branches)
                best_result = {'branches': current_branches, 'paths': [list(paths[k]) for k in paths]}
                break

    start_time = time.time()
    deadline = start_time + 85

    while time.time() < deadline:
        target_size = best_size + 1
        if target_size > n:
            break
        
        found = False
        for _ in range(2000):
            if time.time() > deadline:
                break
            random_branches = random.sample(range(n), target_size)
            for _ in range(2):
                paths = check_subdivision(random_branches)
                if paths:
                    best_size = target_size
                    best_result = {'branches': random_branches, 'paths': [list(paths[k]) for k in paths]}
                    found = True
                    break
            if found:
                break
                
    if best_result['branches'] is None or len(best_result['branches']) < 2:
        u, v = edges[0]
        best_result = {'branches': [u, v], 'paths': [[u, v]]}
        
    branches = [int(b) for b in best_result['branches']]
    paths = [[int(v) for v in p] for p in best_result['paths']]
    
    return {'branches': branches, 'paths': paths}
