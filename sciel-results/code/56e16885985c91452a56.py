import time
import random
import sys
from collections import deque
from scipy.optimize import linear_sum_assignment

sys.setrecursionlimit(10000)

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find largest connected component
    visited = [False] * n
    comp_sizes = []
    for i in range(n):
        if not visited[i]:
            stack = [i]
            visited[i] = True
            comp = []
            while stack:
                node = stack.pop()
                comp.append(node)
                for nei in adj[node]:
                    if not visited[nei]:
                        visited[nei] = True
                        stack.append(nei)
            comp_sizes.append((len(comp), comp))
            
    if not comp_sizes:
        return [0]
        
    comp_sizes.sort(reverse=True)
    largest_comp = comp_sizes[0][1]
    comp_set = set(largest_comp)
    comp_size = len(largest_comp)
    
    start_time = time.time()
    time_limit = 85.0  # Conservative margin before 90s kill
    
    best_path = []
    
    # Helper: check if a path is valid
    def is_valid_path(path):
        if not path:
            return False
        if len(path) != len(set(path)):
            return False
        for i in range(len(path) - 1):
            if path[i+1] not in adj[path[i]]:
                return False
        return True
    
    # Helper: try to extend a path from both ends
    def extend_path(path, visited_set, max_extend=50):
        """Try to extend a path greedily from both ends"""
        if not path:
            return path
        
        path = list(path)
        visited_set = set(path)
        improved = True
        count = 0
        
        while improved and count < max_extend:
            improved = False
            count += 1
            
            # Try to extend from the start
            if path:
                start_node = path[0]
                candidates = [nei for nei in adj[start_node] if nei not in visited_set]
                if candidates:
                    # Pick the candidate with fewest unvisited neighbors (Warnsdorff)
                    best_cand = min(candidates, key=lambda x: len([y for y in adj[x] if y not in visited_set]))
                    path.insert(0, best_cand)
                    visited_set.add(best_cand)
                    improved = True
            
            # Try to extend from the end
            if path:
                end_node = path[-1]
                candidates = [nei for nei in adj[end_node] if nei not in visited_set]
                if candidates:
                    best_cand = min(candidates, key=lambda x: len([y for y in adj[x] if y not in visited_set]))
                    path.append(best_cand)
                    visited_set.add(best_cand)
                    improved = True
        
        return path
    
    # Helper: local search - try to improve path by insertion and 2-opt
    def local_search(path, visited_set, max_iters=1000):
        """Try to improve the path using local moves"""
        if len(path) < 2:
            return path
        
        path = list(path)
        best_path = path[:]
        best_len = len(path)
        visited_set = set(path)
        rng = random.Random(42)
        
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            
            improved = False
            n_path = len(path)
            
            # Try to find a vertex not in path that can be inserted
            unvisited = [v for v in comp_set if v not in visited_set]
            
            if unvisited:
                # Try inserting each unvisited vertex at each possible position
                best_insert = None
                best_pos = -1
                
                for v in unvisited:
                    # Try inserting between path[i] and path[i+1]
                    for i in range(n_path - 1):
                        if v in adj[path[i]] and v in adj[path[i+1]]:
                            best_insert = v
                            best_pos = i + 1
                            break
                    if best_insert is not None:
                        break
                    # Try at beginning
                    if v in adj[path[0]]:
                        best_insert = v
                        best_pos = 0
                        break
                    # Try at end
                    if v in adj[path[-1]]:
                        best_insert = v
                        best_pos = n_path
                        break
                
                if best_insert is not None:
                    path.insert(best_pos, best_insert)
                    visited_set.add(best_insert)
                    improved = True
                    n_path = len(path)
                    if n_path > best_len:
                        best_len = n_path
                        best_path = path[:]
            
            # Try 2-opt move that might allow further insertions
            if not improved and n_path >= 4:
                found_2opt = False
                for i in range(n_path - 2):
                    if found_2opt:
                        break
                    for j in range(i + 2, n_path):
                        # Check if reversing segment [i+1..j] creates valid edges
                        if path[j] not in adj[path[i]]:
                            continue
                        
                        if j < n_path - 1:
                            if path[j-1] not in adj[path[i+1]]:
                                continue
                        
                        # Perform the 2-opt
                        new_path = path[:i+1] + path[i+1:j+1][::-1] + path[j+1:]
                        
                        # Validate
                        valid = True
                        for k in range(len(new_path) - 1):
                            if new_path[k+1] not in adj[new_path[k]]:
                                valid = False
                                break
                        
                        if valid:
                            path = new_path
                            found_2opt = True
                            improved = True
                            break
            
            if not improved:
                break
        
        return best_path if len(best_path) >= len(path) else path
    
    # DFS with Warnsdorff's rule (iterative to avoid recursion depth issues)
    def dfs_iterative(start_node, comp_set, adj, time_limit, start_time):
        """Iterative DFS with Warnsdorff's rule"""
        best_path = []
        stack = [(start_node, [start_node], {start_node})]
        
        while stack:
            if time.time() - start_time > time_limit:
                break
                
            node, path, visited_set = stack.pop()
            
            if len(path) > len(best_path):
                best_path = path[:]
                
            if len(path) == len(comp_set):
                return best_path  # Found Hamiltonian path!
                
            # Get unvisited neighbors
            candidates = [nei for nei in adj[node] if nei not in visited_set]
            
            if not candidates:
                continue
                
            # Warnsdorff's rule: sort by number of unvisited neighbors
            def score(nei):
                unvisited_nei = len([x for x in adj[nei] if x not in visited_set])
                return (unvisited_nei, nei)
            
            candidates.sort(key=score)
            
            # Push neighbors in reverse order so first candidate is processed first
            for nei in reversed(candidates):
                new_visited = visited_set | {nei}
                new_path = path + [nei]
                stack.append((nei, new_path, new_visited))
        
        return best_path
    
    # Strategy 0: Bipartite matching-based initialization
    # Create a bipartite graph where each vertex v has v_in and v_out
    # Edge from u_out to v_in exists if (u,v) is an edge in original graph
    # Find maximum matching, then extract path from matching structure
    def matching_based_path(comp_set, adj, n):
        """Use maximum bipartite matching to find a long path"""
        if len(comp_set) <= 2:
            return list(comp_set)
        
        # Build cost matrix for Hungarian algorithm
        # We want to maximize the number of matched edges in a bipartite graph
        # Left side: vertices 0..n-1 (as "out" nodes)
        # Right side: vertices 0..n-1 (as "in" nodes)
        # Cost[u][v] = 0 if (u,v) is an edge, else large penalty
        
        INF = 10**9
        cost = [[INF] * n for _ in range(n)]
        
        for u in comp_set:
            for v in adj[u]:
                if v in comp_set:
                    cost[u][v] = 0
        
        # Use Hungarian algorithm to find minimum cost matching
        # (equivalent to maximum cardinality matching with 0/INF costs)
        try:
            row_ind, col_ind = linear_sum_assignment(cost)
        except:
            return list(comp_set)[:2]
        
        # Extract matched edges
        matched_edges = []
        for u, v in zip(row_ind, col_ind):
            if cost[u][v] == 0:  # Only keep valid edges
                matched_edges.append((u, v))
        
        if not matched_edges:
            return list(comp_set)[:2]
        
        # Build a directed graph from matched edges
        # Each vertex has at most one outgoing and one incoming edge in the matching
        # Find the longest path in this functional graph
        
        out_map = {}
        in_map = {}
        for u, v in matched_edges:
            out_map[u] = v
            in_map[v] = u
        
        # Find all paths in the functional graph
        visited = set()
        best_path = []
        
        for start in comp_set:
            if start in visited:
                continue
            
            # Walk forward from start
            path = [start]
            current = start
            visited.add(current)
            
            while current in out_map:
                next_node = out_map[current]
                if next_node in visited:
                    break
                path.append(next_node)
                visited.add(next_node)
                current = next_node
            
            if len(path) > len(best_path):
                best_path = path[:]
        
        return best_path if best_path else list(comp_set)[:2]
    
    # Try matching-based initialization
    if time.time() - start_time < time_limit - 10:
        match_path = matching_based_path(comp_set, adj, n)
        if match_path and is_valid_path(match_path):
            if len(match_path) > len(best_path):
                best_path = match_path[:]
            
            # Extend the matching-based path
            if best_path:
                visited_set = set(best_path)
                extended = extend_path(best_path, visited_set, max_extend=100)
                if len(extended) > len(best_path):
                    best_path = extended
                    visited_set = set(best_path)
                
                # Local search on matching-based path
                if time.time() - start_time < time_limit - 5:
                    polished = local_search(best_path, visited_set, max_iters=500)
                    if len(polished) > len(best_path):
                        best_path = polished
    
    # Strategy 1: Try DFS from various starting nodes
    nodes_to_try = largest_comp[:]
    rng = random.Random(12345)
    rng.shuffle(nodes_to_try)
    
    # Also try nodes sorted by degree (low degree first for Warnsdorff)
    degree_sorted = sorted(largest_comp, key=lambda x: len(adj[x]))
    
    # Combine: low degree nodes first, then shuffled
    start_nodes = degree_sorted[:10] + nodes_to_try[:20]
    # Remove duplicates while preserving order
    seen = set()
    unique_starts = []
    for node in start_nodes:
        if node not in seen:
            seen.add(node)
            unique_starts.append(node)
    
    for start_node in unique_starts:
        if time.time() - start_time > time_limit:
            break
            
        path = dfs_iterative(start_node, comp_set, adj, time_limit, start_time)
        
        if len(path) > len(best_path):
            best_path = path[:]
            
        # Try to extend the found path
        if best_path:
            visited_set = set(best_path)
            extended = extend_path(best_path, visited_set, max_extend=100)
            if len(extended) > len(best_path):
                best_path = extended
                visited_set = set(best_path)
            
            # Try local search
            if time.time() - start_time < time_limit - 5:
                polished = local_search(best_path, visited_set, max_iters=500)
                if len(polished) > len(best_path):
                    best_path = polished
        
        if len(best_path) == comp_size:
            break
    
    # Strategy 2: Greedy construction with random restarts
    for _ in range(50):
        if time.time() - start_time > time_limit:
            break
        
        # Random start
        start_node = rng.choice(largest_comp)
        path = [start_node]
        visited_set = {start_node}
        
        while True:
            node = path[-1]
            candidates = [nei for nei in adj[node] if nei not in visited_set]
            if not candidates:
                break
            
            # Warnsdorff: pick neighbor with fewest unvisited neighbors
            best_cand = min(candidates, key=lambda x: len([y for y in adj[x] if y not in visited_set]))
            path.append(best_cand)
            visited_set.add(best_cand)
        
        if len(path) > len(best_path):
            best_path = path[:]
            visited_set = set(best_path)
        
        # Extend from both ends
        extended = extend_path(best_path, visited_set, max_extend=100)
        if len(extended) > len(best_path):
            best_path = extended
            visited_set = set(best_path)
        
        # Local search
        if time.time() - start_time < time_limit - 5:
            polished = local_search(best_path, visited_set, max_iters=500)
            if len(polished) > len(best_path):
                best_path = polished
    
    # Strategy 3: Combined 2-opt + insertion local search
    def combined_local_search(path, max_iters=5000):
        """Apply 2-opt moves and vertex insertions to improve the path"""
        if len(path) < 3:
            return path
        
        path = list(path)
        best_path = path[:]
        best_len = len(path)
        visited_set = set(path)
        rng = random.Random(99)
        
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            
            improved = False
            n_path = len(path)
            
            # Phase 1: Try to insert unvisited vertices
            unvisited = [v for v in comp_set if v not in visited_set]
            
            if unvisited:
                # Sort unvisited by degree (prefer lower degree for Warnsdorff-like behavior)
                unvisited.sort(key=lambda x: len(adj[x]))
                
                for v in unvisited[:30]:  # Limit candidates for speed
                    # Try inserting between path[i] and path[i+1]
                    inserted = False
                    for i in range(n_path - 1):
                        if v in adj[path[i]] and v in adj[path[i+1]]:
                            path.insert(i + 1, v)
                            visited_set.add(v)
                            n_path += 1
                            improved = True
                            inserted = True
                            if n_path > best_len:
                                best_len = n_path
                                best_path = path[:]
                            break
                    
                    if not inserted and not improved:
                        # Try at beginning
                        if v in adj[path[0]]:
                            path.insert(0, v)
                            visited_set.add(v)
                            n_path += 1
                            improved = True
                            if n_path > best_len:
                                best_len = n_path
                                best_path = path[:]
                        # Try at end
                        elif v in adj[path[-1]]:
                            path.append(v)
                            visited_set.add(v)
                            n_path += 1
                            improved = True
                            if n_path > best_len:
                                best_len = n_path
                                best_path = path[:]
                    
                    if improved:
                        break
            
            # Phase 2: Try 2-opt moves
            if not improved and n_path >= 4:
                found_2opt = False
                # Try all 2-opt moves for thoroughness
                for i in range(n_path - 2):
                    if found_2opt:
                        break
                    for j in range(i + 2, n_path):
                        if path[j] not in adj[path[i]]:
                            continue
                        
                        if j < n_path - 1:
                            if path[j-1] not in adj[path[i+1]]:
                                continue
                        
                        # Perform the 2-opt
                        new_path = path[:i+1] + path[i+1:j+1][::-1] + path[j+1:]
                        
                        # Validate
                        valid = True
                        for k in range(len(new_path) - 1):
                            if new_path[k+1] not in adj[new_path[k]]:
                                valid = False
                                break
                        
                        if valid:
                            path = new_path
                            found_2opt = True
                            improved = True
                            break
            
            # Phase 3: Try 3-opt moves (reverse two segments)
            if not improved and n_path >= 6:
                found_3opt = False
                for i in range(n_path - 4):
                    if found_3opt:
                        break
                    for j in range(i + 2, n_path - 1):
                        if found_3opt:
                            break
                        for k in range(j + 2, n_path):
                            # Reverse segments [i+1..j] and [j+1..k]
                            # New edges: path[i]-path[j], path[j-1]-path[j+1] (if j<n-1), path[k-1]-path[i+1]
                            if path[j] not in adj[path[i]]:
                                continue
                            
                            if j < n_path - 1:
                                if path[j-1] not in adj[path[j+1]]:
                                    continue
                            
                            if k < n_path - 1:
                                if path[k-1] not in adj[path[i+1]]:
                                    continue
                            
                            new_path = path[:i+1] + path[i+1:j+1][::-1] + path[j+1:k+1][::-1] + path[k+1:]
                            
                            # Validate
                            valid = True
                            for idx in range(len(new_path) - 1):
                                if new_path[idx+1] not in adj[new_path[idx]]:
                                    valid = False
                                    break
                            
                            if valid:
                                path = new_path
                                found_3opt = True
                                improved = True
                                break
            
            if not improved:
                break
        
        return best_path
    
    # Apply combined local search to the best path
    if best_path and len(best_path) > 3:
        if time.time() - start_time < time_limit - 5:
            combined_result = combined_local_search(best_path, max_iters=5000)
            if len(combined_result) > len(best_path):
                best_path = combined_result
    
    # Final aggressive polish with multiple passes
    if best_path:
        for polish_pass in range(5):
            if time.time() - start_time > time_limit - 2:
                break
            visited_set = set(best_path)
            polished = local_search(best_path, visited_set, max_iters=1500)
            if len(polished) > len(best_path):
                best_path = polished
            else:
                break
        
        # One more combined search pass
        if time.time() - start_time < time_limit - 1 and len(best_path) > 3:
            final_combined = combined_local_search(best_path, max_iters=3000)
            if len(final_combined) > len(best_path):
                best_path = final_combined
        
        # Final local search pass
        if time.time() - start_time < time_limit - 0.5:
            visited_set = set(best_path)
            final_polish = local_search(best_path, visited_set, max_iters=1000)
            if len(final_polish) > len(best_path):
                best_path = final_polish
    
    # Validate final path
    if not is_valid_path(best_path):
        # Fallback: return a valid path
        best_path = [largest_comp[0]]
    
    # Ensure path is nonempty
    if not best_path:
        best_path = [0]
    
    return best_path
