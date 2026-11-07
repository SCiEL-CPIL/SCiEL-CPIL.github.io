import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list and set for O(1) lookup
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Convert to sorted lists for consistent iteration
    adj_list = [sorted(adj[i]) for i in range(n)]
    
    # Calculate degrees
    degrees = [len(adj_list[i]) for i in range(n)]
    
    # Global deadline: 90s limit, use 85s for safety
    start_time = time.time()
    global_deadline = start_time + 85.0

    def find_paths(u, v, forbidden, max_len=10, max_count=20):
        """
        Find simple paths from u to v avoiding 'forbidden' internal nodes.
        Returns a list of paths (lists of vertices), shortest first.
        """
        if u == v:
            return []
        
        # BFS to find shortest paths
        # We want to find multiple shortest paths if possible, then slightly longer ones
        # To keep it fast, we primarily look for shortest paths.
        
        paths = []
        
        # Standard BFS
        visited = {u}
        queue = deque()
        queue.append((u, [u]))
        
        found_depth = -1
        
        while queue:
            current, path = queue.popleft()
            
            if current == v:
                if len(path) > 1:
                    paths.append(path)
                if found_depth == -1:
                    found_depth = len(path)
                # If we have enough paths at this depth, we can stop if we are only interested in shortest
                # But we might need alternatives if they conflict later. 
                # For now, collect all shortest paths.
                if len(paths) >= max_count:
                    return paths
                continue
            
            if found_depth != -1 and len(path) >= found_depth:
                # We have found shortest paths, no need to go deeper if we have enough
                # Actually, if we found one, all subsequent BFS nodes at this depth or deeper are longer or equal.
                # If we want ONLY shortest, we stop expanding nodes at depth >= found_depth.
                # But we need to finish the current layer to get all shortest paths.
                # Since we append to paths when we hit v, and BFS explores by depth,
                # once we hit v, we know the shortest length.
                # We should not expand neighbors of nodes at depth == found_depth because they would be longer.
                # But we are at `current` which is NOT v.
                # If len(path) == found_depth, neighbors would make it found_depth+1.
                # So we can prune.
                if len(path) >= found_depth:
                    continue

            for neighbor in adj_list[current]:
                if neighbor == v:
                    # Direct edge to target
                    new_path = path + [neighbor]
                    if found_depth == -1 or len(new_path) <= found_depth:
                         paths.append(new_path)
                         if found_depth == -1:
                             found_depth = len(new_path)
                    # Do not add v to queue to avoid cycles or further expansion from v
                    continue
                if neighbor not in visited and neighbor not in forbidden:
                    visited.add(neighbor)
                    new_path = path + [neighbor]
                    # Heuristic: don't go too deep
                    if len(new_path) < max_len:
                        queue.append((neighbor, new_path))
            
            # If we have found some shortest paths and processed all nodes at that depth, we can stop
            # BFS doesn't strictly track "layer done" easily in this implementation without restructuring.
            # However, the `if len(path) >= found_depth: continue` logic helps.
            # Let's add a break condition if we have enough paths and we are deep enough.
            if len(paths) >= max_count:
                break

        if paths:
            # Sort by length just in case
            paths.sort(key=len)
            return paths[:max_count]
        
        # If no short path found via BFS (maybe blocked), try DFS for longer paths
        # This is expensive, so limit depth and count
        paths = []
        
        def dfs(current, target, path, visited_set, depth):
            if len(paths) >= max_count:
                return
            if depth > max_len:
                return
            if current == target:
                if len(path) > 1:
                    paths.append(list(path))
                return
            
            for neighbor in adj_list[current]:
                if neighbor == target:
                    dfs(neighbor, target, path + [neighbor], visited_set | {neighbor}, depth + 1)
                elif neighbor not in visited_set and neighbor not in forbidden:
                    dfs(neighbor, target, path + [neighbor], visited_set | {neighbor}, depth + 1)
        
        dfs(u, v, [u], {u}, 0)
        paths.sort(key=len)
        return paths[:max_count]

    def check_clique_subdivision(branches):
        """
        Check if the given set of branches supports a clique subdivision.
        Returns list of paths if found, else None.
        """
        k = len(branches)
        if k < 2:
            return None
        
        branch_set = set(branches)
        pairs = list(combinations(branches, 2))
        
        # Pre-compute a "difficulty" score for each pair to order them
        # Pairs with fewer common neighbors are harder
        def pair_score(pair):
            u, v = pair
            # If directly connected, it's "easy" (path length 2) but uses no internal nodes.
            # Actually, direct edge is best.
            if u in adj[v]:
                return 0
            common = len(adj[u] & adj[v])
            return -common # Fewer common neighbors -> more negative -> higher priority in sorted list if we sort ascending? 
                           # We want hardest first. So we want low score first.
                           # Let's say score = number of available short paths.
                           # For now, just use -common.
        
        # Sort pairs: Hardest first (fewest options)
        # We will dynamically re-evaluate or just use a static order based on graph structure
        pairs.sort(key=pair_score)
        
        best_result = None
        local_deadline = min(time.time() + 1.0, global_deadline) # Tight local deadline per subset
        
        # Backtracking state
        # assigned: dict (u,v) -> path
        # used_internal: set of vertices used in interiors
        # idx: index in pairs list
        
        def solve(idx, used_internal, current_paths_dict):
            nonlocal best_result
            if time.time() > local_deadline:
                return False
            
            if idx == len(pairs):
                # All pairs assigned
                # Convert dict to list of paths for return
                # The order of paths in the return value doesn't matter, just that they exist
                best_result = list(current_paths_dict.values())
                return True
            
            u, v = pairs[idx]
            
            # Forbidden nodes: used_internal + all branches except u and v
            forbidden = used_internal | (branch_set - {u, v})
            
            candidates = find_paths(u, v, forbidden, max_len=8, max_count=10)
            
            if not candidates:
                return False
            
            # Try candidates
            # Heuristic: try shortest first
            for path in candidates:
                internal_nodes = set(path[1:-1])
                
                # Verify path is valid (adjacency)
                valid = True
                for i in range(len(path) - 1):
                    if path[i+1] not in adj[path[i]]:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check internal nodes don't conflict
                if internal_nodes & used_internal:
                    continue
                
                new_used_internal = used_internal | internal_nodes
                new_paths_dict = dict(current_paths_dict)
                new_paths_dict[(u, v)] = path
                
                if solve(idx + 1, new_used_internal, new_paths_dict):
                    return True
            
            return False
        
        if solve(0, set(), {}):
            return best_result
        
        return None

    # Candidate Generation
    # Rank vertices by degree
    sorted_vertices = sorted(range(n), key=lambda i: degrees[i], reverse=True)
    
    # We will try to find the maximum k.
    # Start with high k and go down.
    # To limit the number of combinations, we restrict the pool of candidates for each k.
    # For large k, the branches must be well-connected, so they are likely high-degree.
    
    best_solution = None
    best_k = 0
    
    # Max possible k is n, but realistically much smaller.
    # Let's try k from 10 down to 2.
    # If we find a solution for k, we can stop (since we want max k).
    
    # To be safe, we might need to look at more than just the top 15 vertices for smaller k.
    # But for k >= 6, top 15 is usually sufficient.
    
    max_k_to_try = 12
    min_k_to_try = 2
    
    # We iterate k from max down to min
    for k in range(max_k_to_try, min_k_to_try - 1, -1):
        if time.time() > global_deadline - 5.0:
            break
        
        # Determine pool size
        # For k=10, C(15,10) = 3003. C(20,10) = 184756. 
        # 3003 checks is fast if each check is fast.
        # If check is ~1ms, 3s.
        # Let's start with pool size 15 for k>=8, 18 for k=7, 20 for k=6.
        
        if k >= 8:
            pool_size = min(15, n)
        elif k == 7:
            pool_size = min(18, n)
        else:
            pool_size = min(20, n)
            
        pool = sorted_vertices[:pool_size]
        
        found = False
        for subset in combinations(pool, k):
            if time.time() > global_deadline - 2.0:
                break
            
            branches = list(subset)
            paths = check_clique_subdivision(branches)
            
            if paths is not None:
                best_solution = {
                    'branches': branches,
                    'paths': paths
                }
                best_k = k
                found = True
                break
        
        if found:
            break
    
    # If we didn't find anything with the restricted pool, try a larger pool for smaller k
    if best_solution is None:
        for k in range(6, min_k_to_try - 1, -1):
            if time.time() > global_deadline - 2.0:
                break
            
            pool = sorted_vertices[:min(24, n)] # All vertices
            # This is expensive for k=6: C(24,6) = 134596. 
            # We might time out. 
            # Let's add a randomization or limit the number of subsets we try if it's too big.
            # For now, let's just try it but with a very tight per-subset deadline.
            
            # To avoid TLE, we can sample or limit.
            # But let's try the first 5000 subsets.
            count = 0
            for subset in combinations(pool, k):
                if time.time() > global_deadline - 1.0:
                    break
                if count > 5000:
                    break
                count += 1
                
                branches = list(subset)
                paths = check_clique_subdivision(branches)
                
                if paths is not None:
                    best_solution = {
                        'branches': branches,
                        'paths': paths
                    }
                    break
            
            if best_solution is not None:
                break

    # Fallback: find any 2 branches with a path
    if best_solution is None:
        # Find an edge
        for u, v in edges:
            if u in adj[v]:
                best_solution = {
                    'branches': [u, v],
                    'paths': [[u, v]]
                }
                break
        
        if best_solution is None:
            # Should not happen for a connected graph
            best_solution = {
                'branches': [0, 1],
                'paths': [[0, 1]]
            }
    
    # Ensure types are correct
    best_solution['branches'] = [int(b) for b in best_solution['branches']]
    best_solution['paths'] = [[int(v) for v in path] for path in best_solution['paths']]
    
    return best_solution

# EVOLVE-BLOCK-END
