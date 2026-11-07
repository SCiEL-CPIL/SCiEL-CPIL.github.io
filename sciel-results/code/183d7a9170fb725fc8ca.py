import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Leave 10s margin
    
    n = P['n']
    adj = {i: set() for i in range(n)}
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Calculate degrees
    degrees = [len(adj[i]) for i in range(n)]
    vertices_by_degree = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    def find_paths_for_pair(u, v, used, max_len=6):
        """Find all simple paths from u to v in G \ used, length <= max_len."""
        paths = []
        # BFS with path tracking
        # State: (current_node, path)
        queue = deque()
        queue.append((u, [u]))
        visited_in_bfs = {u}
        
        while queue:
            curr, path = queue.popleft()
            if curr == v:
                if len(path) <= max_len + 1: # number of vertices <= max_len + 1
                    paths.append(path)
                continue
            if len(path) > max_len + 1:
                continue
            
            for neighbor in adj[curr]:
                if neighbor not in used and neighbor not in visited_in_bfs:
                    visited_in_bfs.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return paths

    def check_subdivision(branches):
        """Check if branches form a clique subdivision. Returns dict or None."""
        k = len(branches)
        if k < 2:
            return None
        
        pairs = list(combinations(branches, 2))
        used = set(branches)
        
        # Store found paths
        found_paths = {}
        
        def backtrack(pairs_left):
            if not pairs_left:
                return True
            
            # Heuristic: pick pair with fewest available paths
            min_count = float('inf')
            best_pair_idx = -1
            best_paths = []
            
            for i, (u, v) in enumerate(pairs_left):
                paths = find_paths_for_pair(u, v, used, max_len=5)
                if len(paths) < min_count:
                    min_count = len(paths)
                    best_pair_idx = i
                    best_paths = paths
                
                if min_count == 0:
                    break
            
            if min_count == 0:
                return False
            
            u, v = pairs_left[best_pair_idx]
            new_pairs = pairs_left[:best_pair_idx] + pairs_left[best_pair_idx+1:]
            
            # Try each path
            for path in best_paths:
                interior = set(path[1:-1])
                if interior & used:
                    continue # Should not happen if find_paths is correct, but safe
                
                used.update(interior)
                found_paths[(u, v)] = path
                
                if backtrack(new_pairs):
                    return True
                
                used.difference_update(interior)
                del found_paths[(u, v)]
            
            return False
        
        if backtrack(pairs):
            # Construct result
            result_paths = []
            for u, v in pairs:
                p = found_paths[(u, v)]
                result_paths.append(p)
            return {'branches': list(branches), 'paths': result_paths}
        return None

    # Try k from high to low
    # We limit the search to reasonable k
    # Max k is n, but realistically much smaller
    for k in range(10, 1, -1):
        if time.time() > deadline:
            break
            
        # Generate candidate sets
        # Start with top k vertices by degree
        candidates = set()
        
        # 1. Top k
        base_set = tuple(sorted(vertices_by_degree[:k]))
        candidates.add(base_set)
        
        # 2. Local search: swap one vertex
        for i in range(k):
            for w in vertices_by_degree[k:]:
                new_set = list(base_set)
                new_set[i] = w
                new_set.sort()
                candidates.add(tuple(new_set))
        
        # 3. Random samples
        import random
        random.seed(42)
        for _ in range(50):
            sample = random.sample(range(n), k)
            sample.sort()
            candidates.add(tuple(sample))
            
        # Check candidates
        for cand in candidates:
            if time.time() > deadline:
                break
            res = check_subdivision(cand)
            if res:
                # Ensure types are correct
                res['branches'] = [int(b) for b in res['branches']]
                res['paths'] = [[int(v) for v in p] for p in res['paths']]
                return res

    # Fallback: Find a clique
    # Bron-Kerbosch or simple search for small n
    def find_max_clique():
        # Simple recursive search
        max_clique = []
        
        def expand(clique, candidates):
            nonlocal max_clique
            if len(clique) > len(max_clique):
                max_clique = clique[:]
            if not candidates:
                return
            for v in list(candidates):
                expand(clique + [v], [w for w in candidates if w in adj[v] and w > v])
                candidates.remove(v)
                
        expand([], list(range(n)))
        return max_clique

    # If no subdivision found, try to find a clique
    # This is a safety net
    # Note: A clique of size k is a valid subdivision with paths of length 1
    clique = find_max_clique()
    if len(clique) >= 2:
        branches = list(clique)
        paths = []
        for u, v in combinations(branches, 2):
            paths.append([u, v])
        return {'branches': branches, 'paths': paths}
    
    # Last resort: 2 branches
    u, v = P['edges'][0]
    return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

# EVOLVE-BLOCK-END
