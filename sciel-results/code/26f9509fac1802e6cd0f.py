import time
import sys
import random

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find connected components
    visited = [False] * n
    components = []
    for i in range(n):
        if not visited[i]:
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for nei in adj[node]:
                    if not visited[nei]:
                        visited[nei] = True
                        stack.append(nei)
            components.append(comp)
    
    # Sort by size descending
    components.sort(key=len, reverse=True)
    
    # We only care about the largest component(s)
    # For a sparse random graph G(100, 0.018), the largest component is likely most of the vertices
    
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin for 90s limit
    
    best_path = []
    best_len = 0
    
    # Warnsdorff-guided DFS with backtracking
    # The key insight: in sparse graphs, greedy gets stuck. Backtracking lets us explore
    # alternative branches when we hit a dead end.
    
    def greedy_extend(path, visited_set, max_depth, time_limit):
        """Greedy extension using Warnsdorff heuristic."""
        nonlocal best_path, best_len
        adj_list = [sorted(adj[i]) for i in range(n)]
        
        current_path = list(path)
        current_visited = set(visited_set)
        
        while len(current_path) < max_depth:
            if time.time() > time_limit:
                break
            last = current_path[-1]
            candidates = [v for v in adj_list[last] if v not in current_visited]
            if not candidates:
                break
            # Warnsdorff: pick neighbor with fewest unvisited neighbors
            candidates.sort(key=lambda v: (len([u for u in adj_list[v] if u not in current_visited]), v))
            next_v = candidates[0]
            current_path.append(next_v)
            current_visited.add(next_v)
            if len(current_path) > best_len:
                best_len = len(current_path)
                best_path = list(current_path)
        
        return current_path
    
    def try_extend_both_ends(path, visited_set, max_depth, time_limit):
        """Try to extend the path from both ends."""
        nonlocal best_path, best_len
        adj_list = [sorted(adj[i]) for i in range(n)]
        
        current_path = list(path)
        current_visited = set(visited_set)
        
        improved = True
        while improved:
            improved = False
            if time.time() > time_limit:
                break
            
            # Try extending from the left end
            if len(current_path) >= 2:
                left_end = current_path[0]
                candidates = [v for v in adj_list[left_end] if v not in current_visited]
                if candidates:
                    candidates.sort(key=lambda v: (len([u for u in adj_list[v] if u not in current_visited]), v))
                    next_v = candidates[0]
                    current_path.insert(0, next_v)
                    current_visited.add(next_v)
                    improved = True
                    if len(current_path) > best_len:
                        best_len = len(current_path)
                        best_path = list(current_path)
            
            # Try extending from the right end
            if len(current_path) >= 2:
                right_end = current_path[-1]
                candidates = [v for v in adj_list[right_end] if v not in current_visited]
                if candidates:
                    candidates.sort(key=lambda v: (len([u for u in adj_list[v] if u not in current_visited]), v))
                    next_v = candidates[0]
                    current_path.append(next_v)
                    current_visited.add(next_v)
                    improved = True
                    if len(current_path) > best_len:
                        best_len = len(current_path)
                        best_path = list(current_path)
        
        return current_path
    
    def two_opt_search(path, visited_set, max_depth, time_limit):
        """Apply 2-opt local search to rearrange the path for better extension opportunities."""
        nonlocal best_path, best_len
        
        current_path = list(path)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            L = len(current_path)
            
            # Try all 2-opt moves: reverse segment [i+1..j]
            # Only need to check edges at boundaries: (i, i+1)->(i, j) and (j, j+1)->(i+1, j+1)
            for i in range(L - 2):
                if time.time() > time_limit:
                    return current_path
                for j in range(i + 2, L):
                    # Check boundary edges after reversal
                    # Before: path[i]-path[i+1] and path[j]-path[j+1] (if j+1 < L)
                    # After:  path[i]-path[j] and path[i+1]-path[j+1] (if j+1 < L)
                    
                    if j + 1 < L:
                        # Need both new edges to exist
                        if current_path[j] in adj[current_path[i]] and current_path[i+1] in adj[current_path[j+1]]:
                            # Perform reversal
                            current_path[i+1:j+1] = current_path[i+1:j+1][::-1]
                            improved = True
                            if len(current_path) > best_len:
                                best_len = len(current_path)
                                best_path = list(current_path)
                            break
                    else:
                        # j is last element, only need path[i]-path[j]
                        if current_path[j] in adj[current_path[i]]:
                            current_path[i+1:j+1] = current_path[i+1:j+1][::-1]
                            improved = True
                            if len(current_path) > best_len:
                                best_len = len(current_path)
                                best_path = list(current_path)
                            break
                if improved:
                    break
        
        return current_path
    
    def dfs_warnsdorff(start, max_depth, time_limit):
        """DFS with Warnsdorff heuristic and backtracking to find longest path."""
        nonlocal best_path, best_len
        
        path = [start]
        visited_set = {start}
        adj_list = [sorted(adj[i]) for i in range(n)]
        
        def extend_from(current_path, current_visited, max_depth, time_limit):
            """Recursive extension with backtracking."""
            nonlocal best_path, best_len
            
            if time.time() > time_limit:
                return
            
            last = current_path[-1]
            candidates = [v for v in adj_list[last] if v not in current_visited]
            
            if not candidates:
                return
            
            # Warnsdorff ordering: prefer vertices with fewer unvisited neighbors
            candidates.sort(key=lambda v: (len([u for u in adj_list[v] if u not in current_visited]), v))
            
            for v in candidates:
                if time.time() > time_limit:
                    return
                
                current_path.append(v)
                current_visited.add(v)
                
                if len(current_path) > best_len:
                    best_len = len(current_path)
                    best_path = list(current_path)
                
                if len(current_path) < max_depth:
                    extend_from(current_path, current_visited, max_depth, time_limit)
                
                # Backtrack
                current_path.pop()
                current_visited.discard(v)
        
        # Start the search
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
        
        if len(path) < max_depth:
            extend_from(path, visited_set, max_depth, time_limit)
        
        return best_path
    
    # Phase 1: Try DFS from each vertex in the largest component
    largest_comp = components[0] if components else []
    
    # Sort starts by degree (higher degree first) for better initial paths
    largest_comp_sorted = sorted(largest_comp, key=lambda v: len(adj[v]), reverse=True)
    
    for start in largest_comp_sorted:
        if time.time() > deadline:
            break
        
        remaining = deadline - time.time()
        if remaining < 0.5:
            break
        
        # Give each start a time budget proportional to remaining time
        n_remaining = len(largest_comp_sorted) - largest_comp_sorted.index(start)
        time_per_start = max(0.1, min(3.0, remaining / max(1, n_remaining)))
        
        dfs_warnsdorff(start, n, time.time() + time_per_start)
    
    # Phase 2: Polish the best path found with iterative improvement
    if time.time() < deadline and best_path:
        # Iteratively apply: extend both ends -> 2-opt -> extend both ends
        for _ in range(10):
            if time.time() > deadline:
                break
            
            # Try both-ends extension
            extended = try_extend_both_ends(best_path, set(best_path), n, deadline)
            if len(extended) > best_len:
                best_len = len(extended)
                best_path = list(extended)
            
            # Try 2-opt to rearrange for better extension
            if time.time() < deadline:
                rearranged = two_opt_search(best_path, set(best_path), n, deadline)
                if len(rearranged) > best_len:
                    best_len = len(rearranged)
                    best_path = list(rearranged)
            
            # After 2-opt, try extending again
            if time.time() < deadline:
                extended2 = try_extend_both_ends(best_path, set(best_path), n, deadline)
                if len(extended2) > best_len:
                    best_len = len(extended2)
                    best_path = list(extended2)
            
            # Check if we've made progress; if not, stop early
            if best_len == n:
                break
    
    # Phase 3: If we still have time, try more DFS from different starts
    if time.time() < deadline and best_len < n:
        in_best = set(best_path)
        candidates = [v for v in largest_comp_sorted if v not in in_best]
        if not candidates:
            candidates = largest_comp_sorted
        
        for start in candidates[:min(len(candidates), 20)]:
            if time.time() > deadline:
                break
            time_budget = min(deadline - time.time() - 0.5, 2.0)
            if time_budget < 0.1:
                break
            dfs_warnsdorff(start, n, time.time() + time_budget)
        
        # Final polish
        if time.time() < deadline and best_path:
            for _ in range(5):
                if time.time() > deadline:
                    break
                extended = try_extend_both_ends(best_path, set(best_path), n, deadline)
                if len(extended) > best_len:
                    best_len = len(extended)
                    best_path = list(extended)
                if time.time() < deadline:
                    rearranged = two_opt_search(best_path, set(best_path), n, deadline)
                    if len(rearranged) > best_len:
                        best_len = len(rearranged)
                        best_path = list(rearranged)
                if time.time() < deadline:
                    extended2 = try_extend_both_ends(best_path, set(best_path), n, deadline)
                    if len(extended2) > best_len:
                        best_len = len(extended2)
                        best_path = list(extended2)
    
    # Ensure we return a valid non-empty path
    if not best_path:
        if components:
            best_path = [components[0][0]]
        else:
            best_path = [0]
    
    return best_path
