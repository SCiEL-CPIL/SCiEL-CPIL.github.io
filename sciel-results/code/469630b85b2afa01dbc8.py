import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = 85.0
    
    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((u, v))
        edge_set.add((v, u))
        
    def time_up():
        return time.time() - start_time > deadline

    def find_shortest_path(u, v, banned):
        """Find shortest simple path from u to v avoiding banned vertices."""
        if u in banned or v in banned:
            return None
        if u == v:
            return [u]
            
        # BFS
        parent = {u: None}
        q = deque([u])
        while q:
            cur = q.popleft()
            if cur == v:
                # Reconstruct path
                path = []
                node = v
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            for nb in sorted(adj[cur]): # Sort for determinism
                if nb not in parent and nb not in banned:
                    parent[nb] = cur
                    q.append(nb)
        return None

    def find_all_shortest_paths(u, v, banned, limit=10):
        """Find up to 'limit' shortest simple paths from u to v avoiding banned."""
        # First find length
        if u in banned or v in banned:
            return []
        
        # BFS to find distance
        dist = {u: 0}
        q = deque([u])
        while q:
            cur = q.popleft()
            if cur == v:
                break
            for nb in adj[cur]:
                if nb not in dist and nb not in banned:
                    dist[nb] = dist[cur] + 1
                    q.append(nb)
        
        if v not in dist:
            return []
        
        target_len = dist[v]
        paths = []
        
        # DFS to find paths of exactly target_len
        def dfs(cur, path):
            if len(paths) >= limit:
                return
            if cur == v:
                if len(path) - 1 == target_len:
                    paths.append(list(path))
                return
            
            # Pruning: if remaining steps needed > available depth, prune
            # Simple heuristic: just go deep
            for nb in sorted(adj[cur]):
                if nb in banned:
                    continue
                if nb in path:
                    continue
                # Check if we can still reach v in remaining steps
                # dist_from_cur_to_v is not precomputed, but we can check dist[nb] if it exists
                # Actually, dist[nb] is distance from u to nb.
                # We need distance from nb to v.
                # Let's just rely on the limit and depth.
                path.append(nb)
                dfs(nb, path)
                path.pop()
                
        dfs(u, [u])
        return paths

    def verify_solution(branches, paths):
        """Verify a candidate solution. Returns True if valid."""
        if len(branches) < 2:
            return False
        
        branch_set = set(branches)
        if len(branch_set) != len(branches):
            return False
            
        # Check pairs
        pairs_found = set()
        used_interiors = set()
        
        for path in paths:
            if len(path) < 2:
                return False
            u, v = path[0], path[-1]
            if u not in branch_set or v not in branch_set:
                return False
            if u == v:
                return False
            
            pair = frozenset((u, v))
            if pair in pairs_found:
                return False
            pairs_found.add(pair)
            
            # Check interiors
            for node in path[1:-1]:
                if node in branch_set:
                    return False
                if node in used_interiors:
                    return False
                used_interiors.add(node)
                
            # Check edges
            for i in range(len(path) - 1):
                if (path[i], path[i+1]) not in edge_set:
                    return False
                    
        # Check all pairs covered
        expected_pairs = set(frozenset(p) for p in combinations(branches, 2))
        if pairs_found != expected_pairs:
            return False
            
        return True

    def solve_for_branches(branches):
        """Try to find disjoint paths for a given set of branches.
        Returns list of paths if successful, else None."""
        if time_up():
            return None
            
        banned = set(branches)
        paths = {}
        # Order pairs by distance to prioritize short paths
        pairs = list(combinations(branches, 2))
        
        def pair_dist(p):
            u, v = p
            # Quick BFS dist
            if u in banned or v in banned:
                return 999
            dist = {u: 0}
            q = deque([u])
            while q:
                cur = q.popleft()
                if cur == v:
                    return dist[cur]
                for nb in adj[cur]:
                    if nb not in dist and nb not in banned:
                        dist[nb] = dist[cur] + 1
                        q.append(nb)
            return 999
            
        pairs.sort(key=pair_dist)
        
        def backtrack(idx, current_paths, current_banned):
            if time_up():
                return False
            if idx == len(pairs):
                return True
                
            u, v = pairs[idx]
            # Find candidate paths
            # We want short paths to minimize interior usage
            # Try shortest first
            candidates = find_all_shortest_paths(u, v, current_banned, limit=20)
            
            # If no shortest, try longer? 
            # For now, just use shortest. If fails, backtrack.
            for path in candidates:
                new_banned = set(current_banned)
                for node in path[1:-1]:
                    new_banned.add(node)
                    
                current_paths.append(path)
                if backtrack(idx + 1, current_paths, new_banned):
                    return True
                current_paths.pop()
                
            return False
            
        current_paths = []
        if backtrack(0, current_paths, banned):
            return current_paths
        return None

    def greedy_build(starting_set=None):
        """Greedy construction starting from a set or empty."""
        if starting_set is None:
            starting_set = []
            
        branches = list(starting_set)
        
        # If empty, find a good starting edge or triangle
        if not branches:
            # Try to find a triangle
            best_tri = None
            for combo in combinations(range(n), 3):
                if time_up():
                    break
                if all((u, v) in edge_set for u, v in combinations(combo, 2)):
                    best_tri = list(combo)
                    break
            if best_tri:
                branches = best_tri
            else:
                # Fallback to edge
                if edges:
                    branches = list(edges[0])
                else:
                    return None
        elif len(branches) == 1:
            # Find neighbor
            nb = list(adj[branches[0]])[0]
            branches.append(nb)
            
        # Verify initial set
        paths = solve_for_branches(branches)
        if paths is None:
            # Try to reduce? Or just return None
            # If we started with a triangle, it should be valid (edges are paths)
            # Unless solve_for_branches failed due to time?
            # For a triangle, paths are just edges.
            if len(branches) == 3:
                return {'branches': branches, 'paths': [[u, v] for u, v in combinations(branches, 2)]}
            return None
            
        current_best = {'branches': branches, 'paths': paths}
        
        # Try to add vertices
        changed = True
        while changed and not time_up():
            changed = False
            # Sort candidates by degree or something?
            # Try all non-branch vertices
            candidates = [v for v in range(n) if v not in branches]
            # Heuristic: try vertices with high degree or close to branches
            # Sort by number of neighbors in branches (descending)
            candidates.sort(key=lambda v: -len(adj[v] & set(branches)))
            
            for v in candidates:
                if time_up():
                    break
                new_branches = branches + [v]
                new_paths = solve_for_branches(new_branches)
                if new_paths is not None:
                    branches = new_branches
                    paths = new_paths
                    current_best = {'branches': branches, 'paths': paths}
                    changed = True
                    break # Restart loop with new set
                    
        return current_best

    # Strategy:
    # 1. Try to build from a K4 if found (dense start)
    # 2. Try to build from a K3
    # 3. Try to build from random edges
    
    best_result = None
    
    # Find K4s
    k4s = []
    for combo in combinations(range(n), 4):
        if time_up():
            break
        if all((u, v) in edge_set for u, v in combinations(combo, 2)):
            k4s.append(list(combo))
            
    # Try each K4
    for k4 in k4s:
        if time_up():
            break
        res = greedy_build(k4)
        if res and (best_result is None or len(res['branches']) > len(best_result['branches'])):
            best_result = res
        if best_result and len(best_result['branches']) >= 10:
            break # Good enough?
            
    # If no K4 worked or not found, try K3s
    if best_result is None or len(best_result['branches']) < 4:
        k3s = []
        for combo in combinations(range(n), 3):
            if time_up():
                break
            if all((u, v) in edge_set for u, v in combinations(combo, 2)):
                k3s.append(list(combo))
                
        # Sort K3s by density? Or just try a few
        # Try all if few, else sample
        for k3 in k3s[:20]:
            if time_up():
                break
            res = greedy_build(k3)
            if res and (best_result is None or len(res['branches']) > len(best_result['branches'])):
                best_result = res
            if best_result and len(best_result['branches']) >= 6:
                break

    # If still nothing, try random edges
    if best_result is None or len(best_result['branches']) < 3:
        import random
        for _ in range(50):
            if time_up():
                break
            u, v = random.choice(edges)
            res = greedy_build([u, v])
            if res and (best_result is None or len(res['branches']) > len(best_result['branches'])):
                best_result = res

    if best_result is None:
        # Fallback
        u, v = edges[0]
        return {'branches': [u, v], 'paths': [[u, v]]}
        
    # Final verification and cleanup
    branches = list(best_result['branches'])
    paths = [list(p) for p in best_result['paths']]
    
    if verify_solution(branches, paths):
        return {'branches': branches, 'paths': paths}
    else:
        # Should not happen, but fallback
        u, v = edges[0]
        return {'branches': [u, v], 'paths': [[u, v]]}
