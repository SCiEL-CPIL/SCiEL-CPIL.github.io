import time
import sys
from itertools import combinations
from collections import deque
import random

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    try:
        start_time = time.time()
        deadline = start_time + 55
        
        n = P['n']
        adj_list = [[] for _ in range(n)]
        adj_set = [set() for _ in range(n)]
        for u, v in P['edges']:
            adj_list[u].append(v)
            adj_list[v].append(u)
            adj_set[u].add(v)
            adj_set[v].add(u)
        
        # Precompute common-neighbor bitmasks
        common_mask = [0] * (n * n)
        for u in range(n):
            for v in range(u + 1, n):
                mask = 0
                for w in adj_list[u]:
                    if w in adj_set[v]:
                        mask |= (1 << w)
                common_mask[u * n + v] = mask
                common_mask[v * n + u] = mask
        
        def time_check():
            return time.time() > deadline
        
        def bfs_path(src, dst, blocked_mask):
            """BFS finding shortest path from src to dst avoiding blocked_mask vertices."""
            if src == dst:
                return [src]
            visited = 1 << src
            parent = {src: -1}
            queue = deque([src])
            while queue:
                node = queue.popleft()
                if node == dst:
                    path = []
                    cur = dst
                    while cur != -1:
                        path.append(cur)
                        cur = parent[cur]
                    path.reverse()
                    return path
                for neighbor in adj_list[node]:
                    if not (visited >> neighbor) & 1 and not (blocked_mask >> neighbor) & 1:
                        visited |= (1 << neighbor)
                        parent[neighbor] = node
                        queue.append(neighbor)
            return None
        
        def find_paths_for_branches(branches):
            """Check if branches can be connected by pairwise disjoint paths.
            Returns dict mapping (u,v) -> path, or None if infeasible."""
            if len(branches) < 2:
                return None
            branch_set = set(branches)
            branch_mask = 0
            for b in branches:
                branch_mask |= (1 << b)
            pairs = list(combinations(branches, 2))
            
            pair_info = {}
            non_direct_pairs = []
            for u, v in pairs:
                if u in adj_set[v]:
                    pair_info[(u, v)] = ('direct', None)
                else:
                    mask = common_mask[u * n + v] & ~branch_mask
                    if mask == 0:
                        return None
                    pair_info[(u, v)] = ('need', mask)
                    non_direct_pairs.append((u, v))
            
            if not non_direct_pairs:
                # All pairs are direct edges
                result_paths = {}
                for pair in pairs:
                    result_paths[pair] = [pair[0], pair[1]]
                return result_paths
            
            # Try to find disjoint internal paths for non-direct pairs
            # Use greedy assignment with backtracking
            used_mask = 0
            assignment = {}
            
            # Sort by number of available common neighbors (most constrained first)
            sorted_non_direct = sorted(non_direct_pairs, key=lambda p: bin(pair_info[p][1]).count('1'))
            
            def backtrack(idx):
                if time_check():
                    return False
                if idx == len(sorted_non_direct):
                    return True
                u, v = sorted_non_direct[idx]
                # Try BFS with current blocked set
                blocked = branch_mask | used_mask
                path = bfs_path(u, v, blocked)
                if path is None:
                    return False
                internal = path[1:-1]
                # Verify all internal vertices are distinct and not branches
                internal_mask = 0
                for w in internal:
                    if (internal_mask >> w) & 1:
                        return False  # Duplicate
                    if (branch_mask >> w) & 1:
                        return False  # Branch vertex
                    internal_mask |= (1 << w)
                
                assignment[(u, v)] = internal
                used_mask |= internal_mask
                if backtrack(idx + 1):
                    return True
                del assignment[(u, v)]
                return False
            
            if not backtrack(0):
                return None
            
            result_paths = {}
            for pair in pairs:
                u, v = pair
                if pair_info[pair][0] == 'direct':
                    result_paths[pair] = [u, v]
                else:
                    internal = assignment[pair]
                    result_paths[pair] = [u] + internal + [v]
            
            return result_paths
        
        def solve():
            best_result = None
            best_k = 0
            
            degree = [len(adj_list[i]) for i in range(n)]
            sorted_vertices = sorted(range(n), key=lambda x: -degree[x])
            
            # Phase 1: Greedy construction
            greedy_branches = []
            for v in sorted_vertices:
                if time_check():
                    break
                test = greedy_branches + [v]
                if find_paths_for_branches(test) is not None:
                    greedy_branches.append(v)
            
            if greedy_branches:
                paths = find_paths_for_branches(greedy_branches)
                if paths is not None:
                    best_k = len(greedy_branches)
                    best_result = {
                        'branches': [int(b) for b in greedy_branches],
                        'paths': [[int(x) for x in paths[p]] for p in sorted(paths.keys())]
                    }
            
            # Phase 2: Local search - try to swap or add vertices
            if best_result and not time_check():
                current = list(best_result['branches'])
                improved = True
                while improved and not time_check():
                    improved = False
                    current_set = set(current)
                    candidates = [v for v in sorted_vertices if v not in current_set]
                    
                    # Try adding vertices
                    for add_v in candidates:
                        if time_check():
                            break
                        trial = current + [add_v]
                        if find_paths_for_branches(trial) is not None:
                            current = trial
                            improved = True
                            if len(current) > best_k:
                                paths = find_paths_for_branches(current)
                                if paths is not None:
                                    best_k = len(current)
                                    best_result = {
                                        'branches': [int(b) for b in current],
                                        'paths': [[int(x) for x in paths[p]] for p in sorted(paths.keys())]
                                    }
                            break
                    
                    # Try swaps
                    for out_b in list(current):
                        if time_check():
                            break
                        for in_b in candidates:
                            if time_check():
                                break
                            if in_b in set(current):
                                continue
                            trial = [b for b in current if b != out_b] + [in_b]
                            if find_paths_for_branches(trial) is not None:
                                current = trial
                                improved = True
                                if len(current) > best_k:
                                    paths = find_paths_for_branches(current)
                                    if paths is not None:
                                        best_k = len(current)
                                        best_result = {
                                            'branches': [int(b) for b in current],
                                            'paths': [[int(x) for x in paths[p]] for p in sorted(paths.keys())]
                                        }
                                break
            
            # Phase 3: Systematic backtracking search
            if not time_check():
                current_branches = []
                
                def backtrack_clique(start_idx):
                    nonlocal best_result, best_k
                    
                    if time_check():
                        return
                    
                    # Update best if we found something larger
                    if len(current_branches) > best_k:
                        paths = find_paths_for_branches(current_branches)
                        if paths is not None:
                            best_result = {
                                'branches': [int(b) for b in current_branches],
                                'paths': [[int(x) for x in paths[p]] for p in sorted(paths.keys())]
                            }
                            best_k = len(current_branches)
                    
                    for i in range(start_idx, n):
                        if time_check():
                            return
                        
                        candidate = sorted_vertices[i]
                        remaining = n - i
                        if len(current_branches) + remaining <= best_k:
                            break
                        
                        current_branches.append(candidate)
                        if find_paths_for_branches(current_branches) is not None:
                            backtrack_clique(i + 1)
                        current_branches.pop()
                
                backtrack_clique(0)
            
            # Phase 4: Final greedy add pass
            if best_result and not time_check():
                best_set = set(best_result['branches'])
                for v in sorted_vertices:
                    if time_check():
                        break
                    if v in best_set:
                        continue
                    trial = list(best_result['branches']) + [v]
                    if find_paths_for_branches(trial) is not None:
                        best_result['branches'] = [int(x) for x in trial]
                        best_set.add(v)
                        best_k = len(trial)
                        paths = find_paths_for_branches(trial)
                        if paths is not None:
                            best_result['paths'] = [[int(x) for x in paths[p]] for p in sorted(paths.keys())]
            
            if best_result is not None:
                return best_result
            
            # Fallback
            for u, v in P['edges']:
                return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
            
            return {'branches': [0, 1], 'paths': [[0, 1]]}
        
        return solve()
    except Exception:
        for u, v in P['edges']:
            return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
        return {'branches': [0, 1], 'paths': [[0, 1]]}
# EVOLVE-BLOCK-END
