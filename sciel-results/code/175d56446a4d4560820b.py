import time
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = P['n']
    edges = P['edges']
    adj = [[] for _ in range(n)]
    adj_set = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        adj_set[u].add(v)
        adj_set[v].add(u)
    
    degrees = [len(adj[i]) for i in range(n)]
    vertices_by_degree = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    # 1. Check for large Max Clique
    max_clique = []
    def bron_kerbosch(R, P_set, X):
        nonlocal max_clique
        if time.time() > deadline:
            return
        if not P_set and not X:
            if len(R) > len(max_clique):
                max_clique = list(R)
            return
        if P_set or X:
            pivot = max(list(P_set) + list(X), key=lambda v: len(adj_set[v] & P_set))
            candidates = [v for v in P_set if v not in adj_set[pivot]]
        else:
            candidates = list(P_set)
            
        for v in candidates:
            bron_kerbosch(R + [v], P_set & adj_set[v], X & adj_set[v])
            P_set.remove(v)
            X.add(v)
            
    bron_kerbosch([], set(range(n)), set())
    
    if len(max_clique) >= 7:
        branches = [int(v) for v in max_clique]
        paths = [[int(u), int(v)] for u, v in combinations(branches, 2)]
        return {'branches': branches, 'paths': paths}
    
    # 2. Search for clique subdivision
    # We try k from 12 down to 3
    # Improved pool sizes and search strategy
    
    def find_subdivision(branches, max_retries=20):
        branches_set = set(branches)
        pairs = list(combinations(branches, 2))
        num_pairs = len(pairs)
        
        # Pre-compute shortest path lengths for sorting pairs
        # We use BFS from each branch to get distances to other branches
        # Note: We must avoid other branches as intermediates
        dist_matrix = [[-1] * n for _ in range(n)]
        for b in branches:
            dist = [-1] * n
            dist[b] = 0
            queue = deque([b])
            while queue:
                curr = queue.popleft()
                for nei in adj[curr]:
                    if dist[nei] == -1:
                        if nei in branches_set and nei != b:
                            continue # Don't go through other branches
                        dist[nei] = dist[curr] + 1
                        queue.append(nei)
            for i in range(n):
                dist_matrix[b][i] = dist[i]
        
        # Sort pairs by shortest path length (ascending)
        # Pairs with shorter paths are "easier" and use fewer vertices
        def get_dist(u, v):
            return dist_matrix[u][v]
        
        sorted_pairs = sorted(pairs, key=lambda p: get_dist(p[0], p[1]))
        
        for attempt in range(max_retries):
            if time.time() > deadline:
                return None
            
            # Shuffle the sorted pairs slightly to break ties and add randomness
            # We keep the general order but shuffle within groups of similar length
            shuffled_pairs = sorted_pairs[:]
            random.shuffle(shuffled_pairs)
            
            used_vertices = set()
            paths = [None] * num_pairs
            pair_to_idx = {p: i for i, p in enumerate(sorted_pairs)} # Map pair to original index in sorted list?
            # Actually, we just need to return the paths. The order of paths in the output doesn't matter, 
            # but we need to ensure we have one for each pair.
            
            def find_paths(idx):
                if time.time() > deadline:
                    return False
                
                if idx == len(shuffled_pairs):
                    return True
                
                u, v = shuffled_pairs[idx]
                
                # BFS to find a path from u to v avoiding used_vertices and other branches
                parent = [-1] * n
                visited = [False] * n
                queue = deque()
                queue.append(u)
                visited[u] = True
                
                found = False
                while queue and not found:
                    curr = queue.popleft()
                    if curr == v:
                        found = True
                        break
                    neighbors = adj[curr][:]
                    random.shuffle(neighbors)
                    for nei in neighbors:
                        if not visited[nei]:
                            if nei in branches_set and nei != v:
                                continue
                            if nei in used_vertices:
                                continue
                            visited[nei] = True
                            parent[nei] = curr
                            queue.append(nei)
                
                if not found:
                    return False
                
                path = []
                curr = v
                while curr != -1:
                    path.append(curr)
                    curr = parent[curr]
                path.reverse()
                
                interior = set(path[1:-1])
                
                if interior & used_vertices:
                    return False
                
                # Store path
                # We need to map back to the correct pair index for the final output
                # Since we just return a list of paths, we can append to a list
                # But we are backtracking, so we need to store in an array indexed by something
                # Let's just store in a list and manage it
                # Actually, the current code stores in `paths` array.
                # Let's use a list `current_paths`
                
                # For simplicity in backtracking, we use a list of paths found so far
                # But we need to backtrack.
                # Let's use a recursive helper that returns a list of paths or None.
                
                # This structure is tricky. Let's stick to the array approach.
                # We need to know which index in the final output this path belongs to.
                # The final output is just a list of paths.
                # So we can just collect them.
                
                # Let's refactor find_paths to return a list of paths for the remaining pairs, or None.
                # No, that's inefficient.
                
                # Let's use the array `paths` indexed by the pair tuple? No, tuples are hashable.
                # Or just a list of paths, and we add/remove.
                
                # Let's use a list `active_paths`
                # If we succeed, we return the list.
                
                # To support backtracking, we can use a list and append/pop.
                
                # Let's try a different structure:
                # find_paths(idx) returns a list of paths for pairs[idx:] or None.
                
                # This is cleaner.
                
                # Re-implementing find_paths with this return type.
                
                # Actually, the previous implementation used a global `paths` array and `used_vertices`.
                # Let's stick to that but be careful.
                
                # Let's use a simple list for paths found in this attempt.
                # If we fail, we discard it.
                
                # But we need to backtrack `used_vertices`.
                
                # Let's use the array approach with a fixed index for each pair in the shuffled order.
                # And we map the pair to its index in the final output?
                # The final output doesn't care about order.
                # So we can just return the list of paths found.
                
                # Let's modify the helper to use a list `paths_list` and append/pop.
                
                # This is getting complicated. Let's stick to the array `paths` indexed by the position in `shuffled_pairs`.
                # And at the end, we just collect all non-None paths.
                
                # We need to map the pair to an index in the `paths` array.
                # Let's just use the index `idx` in the recursion.
                # But `shuffled_pairs` is fixed for this attempt.
                # So `paths[idx]` corresponds to `shuffled_pairs[idx]`.
                
                # This works.
                
                paths[idx] = path
                used_vertices.update(interior)
                
                if find_paths(idx + 1):
                    return True
                
                # Backtrack
                used_vertices.difference_update(interior)
                paths[idx] = None
                return False
            
            if find_paths(0):
                result_paths = [p for p in paths if p is not None]
                return [list(map(int, p)) for p in result_paths]
            
        return None
    
    best_result = None
    
    for k in range(12, 2, -1):
        if time.time() > deadline:
            break
            
        # Pool size: k + 8
        pool_size = min(n, k + 8)
        candidates = vertices_by_degree[:pool_size]
        
        # Filter candidates by degree >= k-1
        # This is a strong heuristic for subdivisions
        candidates = [v for v in candidates if degrees[v] >= k - 1]
        
        if len(candidates) < k:
            continue
            
        # Limit number of combinations to check to stay within time
        # If too many, we might need to be smarter.
        # For k=12, pool=20, C(20,12) = 125970. Too many.
        # We should reduce pool size or use random sampling.
        
        # Let's use random sampling if combinations are too many.
        num_combos = len(list(combinations(candidates, k))) # This is expensive to compute
        # Approximate
        if len(candidates) > k + 10:
            # Sample some combinations
            num_samples = 500
            combos = [tuple(random.sample(candidates, k)) for _ in range(num_samples)]
        else:
            combos = list(combinations(candidates, k))
            
        for combo in combos:
            if time.time() > deadline:
                break
                
            branches = list(combo)
            # Ensure branches are unique and sorted for consistency?
            # Not necessary, but good for debugging.
            
            paths = find_subdivision(branches, max_retries=5)
            
            if paths:
                return {
                    'branches': [int(b) for b in branches],
                    'paths': paths
                }
                
    # Fallback: Max Clique
    if max_clique:
        branches = [int(v) for v in max_clique]
        paths = [[int(u), int(v)] for u, v in combinations(branches, 2)]
        return {'branches': branches, 'paths': paths}
    
    # Fallback: Any triangle
    es = {tuple(e) for e in edges}
    for row in combinations(range(n), 3):
        if all(e in es for e in combinations(row, 2)):
            return {
                'branches': [int(u) for u in row],
                'paths': [list(map(int, e)) for e in combinations(row, 2)]
            }
    
    # Absolute fallback
    u, v = edges[0]
    return {
        'branches': [int(u), int(v)],
        'paths': [[int(u), int(v)]]
    }
