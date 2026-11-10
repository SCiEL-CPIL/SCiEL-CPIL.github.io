import time
import sys
from itertools import combinations
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute degrees for heuristics
    degrees = [len(adj[i]) for i in range(n)]
    
    def find_short_paths(src, dst, forbidden, max_len=4):
        """Find all simple paths from src to dst avoiding forbidden vertices, length <= max_len."""
        paths = []
        # BFS/DFS
        # State: (current, path, visited)
        stack = [(src, [src], {src})]
        while stack:
            curr, path, visited = stack.pop()
            if curr == dst:
                if len(path) - 1 <= max_len:
                    paths.append(path)
                continue
            if len(path) - 1 >= max_len:
                continue
            for nb in adj[curr]:
                if nb not in visited and nb not in forbidden:
                    stack.append((nb, path + [nb], visited | {nb}))
        return paths

    def check_clique(B, max_path_len=4, time_limit=0.05):
        """
        Check if B forms a clique subdivision.
        Returns dict with paths if valid, else None.
        """
        start = time.time()
        B_set = set(B)
        pairs = list(combinations(B, 2))
        
        # 1. Find candidate paths for each pair
        # Forbidden for path between u,v is B \ {u,v}
        candidate_paths = {}
        for u, v in pairs:
            forbidden = B_set - {u, v}
            paths = find_short_paths(u, v, forbidden, max_path_len)
            if not paths:
                return None
            # Sort by length (shorter is better to save internal vertices)
            paths.sort(key=len)
            candidate_paths[(u, v)] = paths
        
        # 2. Backtracking to assign disjoint paths
        # Order pairs by number of candidates (most constrained first)
        pairs_sorted = sorted(pairs, key=lambda p: len(candidate_paths[p]))
        
        assignment = {}
        used_internals = set()
        
        def backtrack(idx):
            if time.time() - start > time_limit:
                return False
            if idx == len(pairs_sorted):
                return True
            
            u, v = pairs_sorted[idx]
            for path in candidate_paths[(u, v)]:
                # Check if internal vertices are disjoint
                internals = set(path[1:-1])
                if not internals.issubset(used_internals) is False and not internals.intersection(used_internals):
                    # Actually: if internals and used_internals are disjoint
                    if not (internals & used_internals):
                        assignment[(u, v)] = path
                        used_internals.update(internals)
                        
                        if backtrack(idx + 1):
                            return True
                        
                        # Undo
                        used_internals.difference_update(internals)
                        del assignment[(u, v)]
            
            return False
        
        if backtrack(0):
            # Construct result
            result_paths = []
            for u, v in pairs:
                result_paths.append(assignment[(u, v)])
            return {'branches': list(B), 'paths': result_paths}
        
        return None

    def solve():
        best_result = None
        best_k = 0
        
        # Try k from high to low
        # We limit the search space by using heuristics for candidate sets
        # 1. Top k by degree
        # 2. Random samples
        # 3. Neighbors of top nodes
        
        vertices_by_degree = sorted(range(n), key=lambda x: degrees[x], reverse=True)
        
        # We will try to find a valid set for k
        for k in range(12, 1, -1): # Try up to 12, likely 7-9 is max
            if time.time() > deadline - 5:
                break
            
            candidates_checked = set()
            found = False
            
            # Strategy 1: Top k by degree
            top_k = tuple(vertices_by_degree[:k])
            if top_k not in candidates_checked:
                candidates_checked.add(top_k)
                res = check_clique(list(top_k))
                if res:
                    return res
            
            # Strategy 2: Random sampling
            # Number of samples depends on time left
            import random
            random.seed(42) # Reproducibility
            
            num_samples = 1000
            for _ in range(num_samples):
                if time.time() > deadline - 2:
                    break
                # Sample k vertices
                # Weighted by degree?
                sample = random.sample(range(n), k)
                sample_tuple = tuple(sorted(sample))
                if sample_tuple in candidates_checked:
                    continue
                candidates_checked.add(sample_tuple)
                
                res = check_clique(list(sample_tuple))
                if res:
                    return res
            
            # Strategy 3: Local search / Swapping
            # Start with top_k, try swapping one vertex
            base = list(vertices_by_degree[:k])
            for i in range(k):
                if time.time() > deadline - 2:
                    break
                for j in range(k, n):
                    if vertices_by_degree[j] == base[i]:
                        continue
                    new_set = base.copy()
                    new_set[i] = vertices_by_degree[j]
                    new_tuple = tuple(sorted(new_set))
                    if new_tuple in candidates_checked:
                        continue
                    candidates_checked.add(new_tuple)
                    
                    res = check_clique(list(new_tuple))
                    if res:
                        return res

        # Fallback: simple edge
        u, v = P['edges'][0]
        return {'branches': [u, v], 'paths': [[u, v]]}
        
    return solve()
# EVOLVE-BLOCK-END
