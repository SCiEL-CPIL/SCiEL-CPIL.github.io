import time
import sys
from itertools import combinations
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s margin for safety
    
    n = P['n']
    # Build adjacency list and sets for fast lookup
    adj_list = [[] for _ in range(n)]
    adj_set = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj_list[u].append(v)
        adj_list[v].append(u)
        adj_set[u].add(v)
        adj_set[v].add(u)
    
    def time_check():
        return time.time() > deadline
    
    def bfs_path(src, dst, blocked):
        """Find a simple path from src to dst avoiding blocked vertices (except src and dst)."""
        if src == dst:
            return [src]
        parent = {src: None}
        queue = deque([src])
        while queue:
            node = queue.popleft()
            if node == dst:
                path = []
                cur = dst
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
            for neighbor in adj_list[node]:
                if neighbor not in parent and neighbor not in blocked:
                    parent[neighbor] = node
                    queue.append(neighbor)
        return None
    
    def find_paths_for_branches(branches):
        """Try to find disjoint internal paths for all pairs of branches.
        Returns dict of pair -> path, or None if infeasible."""
        branch_set = set(branches)
        pairs = list(combinations(branches, 2))
        
        # Identify direct edges and compute options for non-direct pairs
        pair_options = {}
        non_direct_pairs = []
        for u, v in pairs:
            if u in adj_set[v]:
                pair_options[(u, v)] = 'direct'
            else:
                common = set()
                for w in adj_list[u]:
                    if w not in branch_set and w in adj_set[v]:
                        common.add(w)
                pair_options[(u, v)] = common
                non_direct_pairs.append((u, v))
        
        # Feasibility check
        for pair in non_direct_pairs:
            if len(pair_options[pair]) == 0:
                return None
        
        # Sort non-direct pairs by number of options (most constrained first)
        non_direct_pairs.sort(key=lambda p: len(pair_options[p]))
        
        assignment = {}
        used_internals = set()
        
        def backtrack(idx):
            if time_check():
                return False
            if idx == len(non_direct_pairs):
                return True
            pair = non_direct_pairs[idx]
            u, v = pair
            
            blocked = branch_set | used_internals
            blocked.discard(u)
            blocked.discard(v)
            
            path = bfs_path(u, v, blocked)
            if path is None:
                return False
            
            internal = path[1:-1]
            
            if len(internal) != len(set(internal)):
                return False
            for w in internal:
                if w in branch_set:
                    return False
            
            for w in internal:
                used_internals.add(w)
            assignment[pair] = internal
            
            if backtrack(idx + 1):
                return True
            
            for w in internal:
                used_internals.discard(w)
            del assignment[pair]
            
            return False
        
        if not backtrack(0):
            return None
        
        result_paths = {}
        for pair in pairs:
            u, v = pair
            if pair_options[pair] == 'direct':
                result_paths[pair] = [u, v]
            else:
                internal = assignment[pair]
                result_paths[pair] = [u] + internal + [v]
        
        return result_paths
    
    def solve():
        """Find maximum number of branches using backtracking clique search."""
        best_result = None
        best_k = 0
        
        # Sort vertices by degree descending for better pruning
        degree = [len(adj_list[i]) for i in range(n)]
        sorted_vertices = sorted(range(n), key=lambda x: -degree[x])
        
        # Backtracking: build branch set incrementally
        current_branches = []
        
        def backtrack_clique(candidate_start_idx):
            nonlocal best_result, best_k
            
            if time_check():
                return
            
            # Update best if current set is valid and larger
            if len(current_branches) > best_k:
                paths = find_paths_for_branches(current_branches)
                if paths is not None:
                    best_result = {
                        'branches': [int(x) for x in current_branches],
                        'paths': [[int(x) for x in p] for p in sorted(paths.values(), key=lambda p: (p[0], p[-1]))]
                    }
                    best_k = len(current_branches)
            
            # Try adding more vertices
            for i in range(candidate_start_idx, n):
                if time_check():
                    return
                
                candidate = sorted_vertices[i]
                
                # Pruning: if we can't beat best_k even with all remaining, skip
                remaining = n - i
                if len(current_branches) + remaining <= best_k:
                    break
                
                # Check if candidate can be added
                if candidate in current_branches:
                    continue
                
                # Try adding candidate
                current_branches.append(candidate)
                
                # Verify feasibility of the new set
                paths = find_paths_for_branches(current_branches)
                if paths is not None:
                    # Feasible! Recurse to try adding more
                    backtrack_clique(i + 1)
                
                current_branches.pop()
        
        # Start backtracking from empty set
        backtrack_clique(0)
        
        if best_result is not None:
            return best_result
        
        # Fallback: find at least 2 branches
        for u, v in P['edges']:
            return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
        
        return {'branches': [0, 1], 'paths': [[0, 1]]}
    
    result = solve()
    
    # Ensure types are correct
    if result is not None:
        result['branches'] = [int(x) for x in result['branches']]
        result['paths'] = [[int(x) for x in p] for p in result['paths']]
    
    return result
# EVOLVE-BLOCK-END
