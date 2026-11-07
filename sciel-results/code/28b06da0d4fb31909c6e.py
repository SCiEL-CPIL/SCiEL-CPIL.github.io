"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    adj_list = [sorted(adj[i]) for i in range(n)]
    degrees = [len(adj_list[i]) for i in range(n)]
    
    global_deadline = time.time() + 85.0
    
    def find_shortest_paths(u, v, forbidden, max_paths=10):
        """Find shortest simple paths from u to v avoiding forbidden internal nodes."""
        if u == v:
            return []
        
        best_len = None
        shortest_paths = []
        
        queue = deque()
        queue.append((u, [u]))
        visited = {u}
        
        found_target = False
        
        while queue and not found_target:
            layer_size = len(queue)
            for _ in range(layer_size):
                current, path = queue.popleft()
                
                if current == v:
                    if len(path) > 1:
                        if best_len is None:
                            best_len = len(path)
                        if len(path) == best_len:
                            shortest_paths.append(path)
                    found_target = True
                    continue
                
                for neighbor in adj_list[current]:
                    if neighbor == v:
                        new_path = path + [neighbor]
                        if best_len is None:
                            best_len = len(new_path)
                        if len(new_path) == best_len:
                            shortest_paths.append(new_path)
                        found_target = True
                    elif neighbor not in visited and neighbor not in forbidden and len(path) < 6:
                        visited.add(neighbor)
                        new_path = path + [neighbor]
                        queue.append((neighbor, new_path))
            
            if found_target:
                break
        
        if shortest_paths:
            shortest_paths.sort(key=len)
            return shortest_paths[:max_paths]
        
        # Fallback: DFS for longer paths
        paths = []
        
        def dfs(current, target, path, visited_set):
            if len(paths) >= max_paths:
                return
            if current == target:
                if len(path) > 1:
                    paths.append(list(path))
                return
            
            for neighbor in adj_list[current]:
                if neighbor == target:
                    dfs(neighbor, target, path + [neighbor], visited_set | {neighbor})
                elif neighbor not in visited_set and neighbor not in forbidden and len(path) < 8:
                    dfs(neighbor, target, path + [neighbor], visited_set | {neighbor})
        
        dfs(u, v, [u], {u})
        paths.sort(key=len)
        return paths[:max_paths]
    
    def check_clique_subdivision(branches, time_budget=2.0):
        """Check if the given set of branches supports a clique subdivision."""
        k = len(branches)
        if k < 2:
            return None
        
        branch_set = set(branches)
        pairs = list(combinations(branches, 2))
        
        # Order pairs by constraint: direct edges first, then fewer common neighbors
        def pair_constraint_score(pair):
            u, v = pair
            if u in adj[v]:
                return 0
            common = len(adj[u] & adj[v])
            return -common
        
        pairs.sort(key=pair_constraint_score)
        
        best_result = None
        local_deadline = time.time() + time_budget
        
        def solve(idx, used_internal, current_paths):
            nonlocal best_result
            if time.time() > local_deadline:
                return False
            
            if idx == len(pairs):
                best_result = list(current_paths)
                return True
            
            u, v = pairs[idx]
            forbidden = used_internal | (branch_set - {u, v})
            
            candidates = find_shortest_paths(u, v, forbidden, max_paths=8)
            
            if not candidates:
                return False
            
            for path in candidates:
                internal_nodes = set(path[1:-1])
                
                if internal_nodes & used_internal:
                    continue
                
                new_used_internal = used_internal | internal_nodes
                new_paths = current_paths + [path]
                
                if solve(idx + 1, new_used_internal, new_paths):
                    return True
            
            return False
        
        if solve(0, set(), []):
            return best_result
        
        return None
    
    # Greedy independent-set style approach:
    # Build a pool of candidate paths for all pairs, then greedily select
    # a set of branches with mutually disjoint paths.
    
    def greedy_clique_subdivision(time_budget=30.0):
        """
        Greedy approach: try to build a large clique subdivision by
        incrementally adding branches and finding paths.
        """
        deadline = time.time() + time_budget
        
        # Start with high-degree vertices as initial branches
        candidate_order = sorted(range(n), key=lambda i: degrees[i], reverse=True)
        
        best_branches = []
        best_paths = []
        
        # Try to build up from small sets
        for start_k in [3, 4, 5, 6, 7, 8]:
            if time.time() > deadline:
                break
            
            # Try different starting sets
            for start_idx in range(min(10, n - start_k + 1)):
                if time.time() > deadline:
                    break
                
                # Initial set: top-k vertices starting from start_idx
                initial = candidate_order[start_idx:start_idx + start_k]
                if len(initial) < start_k:
                    continue
                
                paths = check_clique_subdivision(initial, time_budget=1.0)
                if paths is None:
                    continue
                
                current_branches = list(initial)
                current_paths = paths
                
                # Try to extend by adding more branches
                for candidate in candidate_order:
                    if time.time() > deadline:
                        break
                    if candidate in current_branches:
                        continue
                    
                    new_branches = current_branches + [candidate]
                    new_paths = check_clique_subdivision(new_branches, time_budget=1.0)
                    if new_paths is not None:
                        current_branches = new_branches
                        current_paths = new_paths
                
                if len(current_branches) > len(best_branches):
                    best_branches = current_branches
                    best_paths = current_paths
        
        if len(best_branches) >= 2:
            return {
                'branches': list(best_branches),
                'paths': [list(p) for p in best_paths]
            }
        
        return None
    
    # Try the greedy approach first
    result = greedy_clique_subdivision(time_budget=60.0)
    
    if result is not None:
        return result
    
    # Fallback: try the original subset-based approach
    candidate_indices = sorted(range(n), key=lambda i: degrees[i], reverse=True)
    
    for k in [8, 7, 6, 5, 4, 3]:
        if time.time() > global_deadline - 5.0:
            break
        
        pool_size = {8: 14, 7: 14, 6: 15, 5: 18, 4: 22, 3: 24}[k]
        candidates = candidate_indices[:pool_size]
        
        from math import comb
        num_subsets = comb(pool_size, k)
        time_per_subset = min((global_deadline - time.time() - 2.0) / max(num_subsets, 1), 2.0)
        time_per_subset = max(time_per_subset, 0.1)
        
        for subset in combinations(candidates, k):
            if time.time() > global_deadline - 2.0:
                break
            branches = list(subset)
            paths = check_clique_subdivision(branches, time_budget=time_per_subset)
            if paths is not None:
                return {
                    'branches': branches,
                    'paths': [list(p) for p in paths]
                }
    
    # Final fallback: 2 branches with a direct edge
    u, v = edges[0]
    return {
        'branches': [u, v],
        'paths': [[u, v]]
    }

# EVOLVE-BLOCK-END
