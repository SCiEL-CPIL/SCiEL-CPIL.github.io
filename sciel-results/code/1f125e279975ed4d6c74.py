import time
import sys
from itertools import combinations
from collections import deque
import random

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin for 90s limit
    
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Helper: Find all simple paths of length <= max_len between u and v
    # avoiding vertices in 'forbidden' (except u and v)
    def find_short_paths(u, v, forbidden, max_len=3):
        paths = []
        # BFS/DFS up to depth max_len
        # Since max_len is small, we can just use recursion
        def dfs(current, target, path, visited):
            if time.time() > deadline:
                return
            if current == target:
                if len(path) - 1 <= max_len:
                    paths.append(list(path))
                return
            
            if len(path) - 1 >= max_len:
                return
                
            for nb in adj[current]:
                if nb in visited:
                    continue
                if nb in forbidden and nb != target:
                    continue
                # Also avoid u if we are not at start? 
                # u is in visited initially, so it's handled.
                visited.add(nb)
                path.append(nb)
                dfs(nb, target, path, visited)
                path.pop()
                visited.remove(nb)
        
        visited = {u}
        dfs(u, v, [u], visited)
        return paths

    def check_subdivision(branches):
        """
        Checks if 'branches' can form a clique subdivision.
        Returns (True, paths_dict) if valid, (False, None) otherwise.
        """
        if len(branches) < 2:
            return True, {}
        
        b_set = set(branches)
        pairs = list(combinations(sorted(branches), 2))
        
        # Pre-compute candidate paths for each pair
        candidate_paths = {}
        for u, v in pairs:
            forbidden = b_set - {u, v}
            # Try to find paths of length 1, 2, 3, 4
            # Start with short, as they use fewer resources
            c_paths = find_short_paths(u, v, forbidden, max_len=4)
            if not c_paths:
                return False, None
            # Sort candidates: shorter paths first, then by interior vertex count (same)
            # Maybe randomize order to help backtracking?
            # For determinism in check, we keep them sorted, but the backtracking
            # will try them in this order.
            candidate_paths[(u, v)] = c_paths
            
        # Order pairs by number of candidates (most constrained first)
        pairs_sorted = sorted(pairs, key=lambda p: len(candidate_paths[p]))
        
        # Backtracking search
        # State: index in pairs_sorted, used_interiors (set), assigned_paths (dict)
        used_interiors = set()
        assigned_paths = {}
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            
            if idx == len(pairs_sorted):
                return True
            
            u, v = pairs_sorted[idx]
            candidates = candidate_paths[(u, v)]
            
            # Try each candidate path
            for path in candidates:
                interiors = set(path[1:-1])
                if interiors & used_interiors:
                    continue
                
                # Accept this path
                used_interiors.update(interiors)
                assigned_paths[(u, v)] = path
                
                if backtrack(idx + 1):
                    return True
                
                # Reject
                used_interiors.difference_update(interiors)
                del assigned_paths[(u, v)]
                
            return False
        
        if backtrack(0):
            return True, assigned_paths
        else:
            return False, None

    # Find Max Clique using Bron-Kerbosch
    best_clique = []
    
    def bronk(R, P_set, X):
        nonlocal best_clique
        if time.time() > deadline:
            return
        if not P_set and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        candidates = list(P_set | X)
        if not candidates:
            return
        pivot = max(candidates, key=lambda u: len(adj[u] & P_set))
        for v in list(P_set - adj[pivot]):
            bronk(R | {v}, P_set & adj[v], X & adj[v])
            P_set = P_set - {v}
            X = X | {v}
            if time.time() > deadline:
                return
    
    bronk(set(), set(range(n)), set())
    
    current_branches = list(best_clique)
    valid, current_paths = check_subdivision(current_branches)
    
    best_size = len(current_branches) if valid else 0
    best_result = None
    if valid:
        best_result = (current_branches, current_paths)
    
    remaining = [v for v in range(n) if v not in set(current_branches)]
    
    # Greedy Extension
    improved = True
    while improved:
        improved = False
        # Shuffle remaining to avoid bias? Or sort by degree?
        # Let's sort by degree descending, but also try random orders in a loop
        for v in sorted(remaining, key=lambda x: len(adj[x]), reverse=True):
            if time.time() > deadline:
                break
            test_branches = current_branches + [v]
            test_valid, test_paths = check_subdivision(test_branches)
            if test_valid:
                current_branches = test_branches
                current_paths = test_paths
                remaining.remove(v)
                improved = True
                if len(current_branches) > best_size:
                    best_size = len(current_branches)
                    best_result = (current_branches, current_paths)
                break # Restart loop with new remaining
    
    # Local Search / Random Restart if stuck
    # If we didn't find a very large set, try random perturbations
    # We have time left?
    if time.time() < deadline - 10:
        current_branches, current_paths = best_result
        remaining = [v for v in range(n) if v not in set(current_branches)]
        
        for _ in range(100): # 100 iterations of local search
            if time.time() > deadline - 5:
                break
            
            # Try to add a random vertex
            if remaining:
                v = random.choice(remaining)
                test_branches = current_branches + [v]
                test_valid, test_paths = check_subdivision(test_branches)
                if test_valid:
                    current_branches = test_branches
                    current_paths = test_paths
                    remaining.remove(v)
                    if len(current_branches) > best_size:
                        best_size = len(current_branches)
                        best_result = (current_branches, current_paths)
                    continue
            
            # If can't add, try to swap
            if len(current_branches) > 2:
                # Remove a random branch
                idx = random.randint(0, len(current_branches) - 1)
                removed = current_branches.pop(idx)
                remaining.append(removed)
                
                # Try to add a new one
                if remaining:
                    v = random.choice(remaining)
                    test_branches = current_branches + [v]
                    test_valid, test_paths = check_subdivision(test_branches)
                    if test_valid:
                        current_branches = test_branches
                        current_paths = test_paths
                        remaining.remove(v)
                        if len(current_branches) > best_size:
                            best_size = len(current_branches)
                            best_result = (current_branches, current_paths)
                    else:
                        # Revert
                        current_branches.append(removed)
                        remaining.remove(removed)
                        # Try another candidate?
                        # Just revert and break to next iteration
                        pass
                else:
                    # Revert
                    current_branches.append(removed)
                    remaining.remove(removed)
            else:
                break
                
    if best_result is None:
        # Fallback to a small valid set, e.g., just an edge
        # Find any edge
        for u, v in P['edges']:
            if u < v:
                best_result = ([u, v], {(u, v): [u, v]})
                break
        if best_result is None:
            # Should not happen if graph has edges
            best_result = ([0, 1], {(0, 1): [0, 1]})

    branches, paths_dict = best_result
    branches = [int(b) for b in sorted(branches)]
    paths_list = []
    for u, v in combinations(sorted(branches), 2):
        p = paths_dict.get((u, v), [u, v])
        paths_list.append([int(x) for x in p])
    
    return {'branches': branches, 'paths': paths_list}
