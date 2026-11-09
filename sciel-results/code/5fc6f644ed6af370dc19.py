import time
import itertools
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    
    n = P['n']
    edges = P['edges']
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Fallback: edge
    if not edges:
        return {'branches': [0, 1], 'paths': [[0, 1]]}
    
    u0, v0 = edges[0]
    best_branches = [u0, v0]
    best_paths = [[u0, v0]]
    
    # Helper to check if a set of branches has a K_k subdivision
    def find_subdivision(branches):
        k = len(branches)
        branch_set = set(branches)
        # Necessary condition: degree >= k-1
        for b in branches:
            if len(adj[b]) < k - 1:
                return None
        
        # Pairs to connect
        pairs = list(itertools.combinations(branches, 2))
        
        # Backtracking search
        # State: used_vertices (set), paths (list of lists), remaining_pairs (list)
        # We want to find a set of paths for all pairs.
        
        # To make it faster, we can precompute available vertices
        avail_vertices = [v for v in range(n) if v not in branch_set]
        
        def dfs(remaining_pairs, used, paths):
            if not remaining_pairs:
                return paths[:]
            
            if time.time() > deadline:
                return None
            
            # Heuristic: pick the pair with the fewest available intermediate vertices
            # Or just pick the first one. Let's try to pick the one with shortest path first.
            # Actually, just picking the first is okay for small k.
            # Let's sort remaining_pairs by some heuristic?
            # For now, take the first pair.
            u, v = remaining_pairs[0]
            rest_pairs = remaining_pairs[1:]
            
            # Find paths from u to v avoiding used and branch_set
            # We can use BFS to find one path.
            # If we need backtracking, we need to find multiple paths.
            # Let's find up to 5 shortest paths.
            
            # BFS for shortest path
            def get_shortest_paths(start, end, forbidden):
                # forbidden is a set of vertices to avoid
                # Returns list of paths (lists of vertices)
                paths_found = []
                visited = {start}
                queue = deque([(start, [start])])
                while queue:
                    if time.time() > deadline:
                        return paths_found
                    node, path = queue.popleft()
                    if node == end:
                        paths_found.append(path)
                        if len(paths_found) >= 5:
                            break
                        continue
                    for nei in adj[node]:
                        if nei not in forbidden and nei not in visited:
                            visited.add(nei)
                            queue.append((nei, path + [nei]))
                return paths_found
            
            forbidden = used | branch_set
            candidate_paths = get_shortest_paths(u, v, forbidden)
            
            for path in candidate_paths:
                new_used = used | set(path[1:-1]) # exclude endpoints
                res = dfs(rest_pairs, new_used, paths + [path])
                if res is not None:
                    return res
            
            return None
        
        return dfs(pairs, set(), [])

    # Try k from 4 up to 10
    # We can also try random subsets if systematic search is too slow
    # But systematic is better for finding the max.
    
    max_k_to_try = 10
    
    for k in range(4, max_k_to_try + 1):
        if time.time() > deadline:
            break
            
        # Filter vertices with degree >= k-1
        candidates = [v for v in range(n) if len(adj[v]) >= k - 1]
        
        # If not enough candidates, skip
        if len(candidates) < k:
            continue
            
        # Iterate combinations
        # To make it faster, we can shuffle or use a heuristic order
        # But let's just iterate.
        for branches in itertools.combinations(candidates, k):
            if time.time() > deadline:
                break
                
            paths = find_subdivision(branches)
            if paths is not None:
                # Valid solution found
                # Convert to correct types
                br = [int(b) for b in branches]
                ps = [[int(p) for p in path] for path in paths]
                
                # Update best
                if k > len(best_branches):
                    best_branches = br
                    best_paths = ps
                    
                # If we found a large one, we might stop early?
                # No, we want the max. But larger k is harder.
                # If we found k=10, it's very good.
                # Let's continue to see if we can find k=11?
                # The loop goes up to max_k_to_try.
                # We can break inner loop if we found one?
                # No, there might be another set of k branches that allows for a more robust solution?
                # The score is just the number of branches. So any valid set of k branches is equally good.
                # So if we find one, we can break the inner loop for this k.
                break
        
        # If we found a solution for this k, should we try k+1?
        # Yes, the loop continues.
        
    # Final check: ensure types are int
    final_branches = [int(b) for b in best_branches]
    final_paths = [[int(p) for p in path] for path in best_paths]
    
    return {'branches': final_branches, 'paths': final_paths}
