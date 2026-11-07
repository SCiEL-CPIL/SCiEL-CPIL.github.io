import time
from itertools import combinations
from collections import deque
import random

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

    # Bron-Kerbosch to find all maximal cliques
    def bron_kerbosch(R, P, X, max_cliques):
        if time_up(): return
        if not P and not X:
            if R:
                max_cliques.append(list(R))
            return
        # Choose pivot u from P union X to maximize branching
        # Here we just pick an arbitrary one for simplicity, or one with max degree in P
        if P:
            pivot = max(P, key=lambda v: len(adj[v] & P))
            candidates = [v for v in P if v not in adj[pivot]]
        else:
            candidates = []
            
        for v in candidates:
            bron_kerbosch(R | {v}, P & adj[v], X & adj[v], max_cliques)
            P.remove(v)
            X.add(v)
            if time_up(): return

    max_cliques = []
    bron_kerbosch(set(), set(range(n)), set(), max_cliques)
    
    # Sort cliques by size descending
    max_cliques.sort(key=len, reverse=True)
    
    best_result = None
    best_size = 0

    def verify_solution(branches, paths):
        if len(branches) < 2:
            return False
        branch_set = set(branches)
        if len(branch_set) != len(branches):
            return False
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
            for node in path[1:-1]:
                if node in branch_set:
                    return False
                if node in used_interiors:
                    return False
                used_interiors.add(node)
            for i in range(len(path) - 1):
                if (path[i], path[i+1]) not in edge_set:
                    return False
        expected_pairs = set(frozenset(p) for p in combinations(branches, 2))
        if pairs_found != expected_pairs:
            return False
        return True

    def update_best(branches, paths):
        nonlocal best_result, best_size
        if verify_solution(branches, paths):
            if len(branches) > best_size:
                best_size = len(branches)
                best_result = {'branches': list(branches), 'paths': [list(p) for p in paths]}

    # Function to try to add a vertex v to a set of branches B
    # Returns (True, new_paths) if successful, (False, None) otherwise
    # new_paths is a dict mapping (u, v) to path
    # We assume B is a clique, so existing paths are edges.
    # But wait, if we have expanded B, existing paths might be longer.
    # So we pass the current paths dict.
    
    def try_add_vertex(v, branches, current_paths):
        """
        Try to add vertex v to branches.
        branches: list of current branch vertices
        current_paths: dict mapping frozenset({u, w}) to path list
        Returns (success, new_paths_dict)
        """
        if time_up(): return False, None
        
        # Identify which pairs (v, u) need new paths
        targets = list(branches)
        
        # Sort targets to make search easier? 
        # Maybe sort by distance from v?
        # Calculate shortest path distance from v to each u in branches, avoiding other branches
        # This helps order the constraints.
        
        def dist_to(v, target, banned):
            if v == target: return 0
            visited = {v}
            q = deque([v])
            dist = {v: 0}
            while q:
                cur = q.popleft()
                for nb in adj[cur]:
                    if nb in banned: continue
                    if nb not in dist:
                        dist[nb] = dist[cur] + 1
                        if nb == target: return dist[nb]
                        q.append(nb)
            return -1
            
        banned_branches = set(branches) # v is not in branches yet
        # We need to avoid all current branches as interiors for the new paths
        # And we need to avoid interiors of existing paths.
        
        existing_interiors = set()
        for path in current_paths.values():
            existing_interiors.update(path[1:-1])
            
        # The new paths must not use any vertex in `banned_branches` (which is `branches`) as interior.
        # They also must not use any vertex in `existing_interiors`.
        # They also must be disjoint from each other.
        
        # Let's collect all forbidden vertices for the interiors of the new paths
        # Forbidden = branches (all of them) U existing_interiors
        # Note: v is the start, so it's allowed as endpoint.
        
        forbidden = set(branches)
        forbidden.update(existing_interiors)
        
        # We need to find paths from v to each u in targets.
        # Let's use backtracking.
        
        new_paths = {}
        used_new_interiors = set()
        
        # Sort targets by distance from v (shorter first) to constrain early
        def get_key(u):
            d = dist_to(v, u, forbidden)
            return d if d != -1 else 1000
            
        targets.sort(key=get_key)
        
        # If any target is unreachable, fail immediately
        for u in targets:
            if dist_to(v, u, forbidden) == -1:
                return False, None

        def backtrack(idx):
            if time_up(): return False
            if idx == len(targets):
                return True
                
            u = targets[idx]
            # Find all simple paths from v to u avoiding forbidden and used_new_interiors
            # We can do a BFS/DFS to find candidates.
            # Given small graph, we can just enumerate simple paths of limited length.
            # Or use a specific DFS.
            
            candidates = []
            # DFS from v to u
            # We want shortest paths first to reduce branching?
            # Let's find all paths of length <= some limit, or just BFS for shortest and then DFS for others?
            # For efficiency, let's just find a few shortest paths.
            
            # BFS to find shortest distance
            dist = {v: 0}
            q = deque([v])
            found_dist = -1
            while q:
                cur = q.popleft()
                if cur == u:
                    found_dist = dist[cur]
                    break
                for nb in adj[cur]:
                    if nb in forbidden or nb in used_new_interiors:
                        continue
                    if nb not in dist:
                        dist[nb] = dist[cur] + 1
                        q.append(nb)
                        
            if found_dist == -1:
                return False
            
            # Now find all paths of length found_dist
            # DFS
            def dfs(cur, path, visited):
                if len(candidates) > 10: # Limit candidates
                    return
                if cur == u:
                    if len(path) - 1 == found_dist:
                        candidates.append(list(path))
                    return
                for nb in adj[cur]:
                    if nb in visited: continue
                    if nb in forbidden or nb in used_new_interiors:
                        continue
                    # Pruning: if current length + 1 > found_dist, skip? 
                    # Actually we are only interested in shortest paths for now to keep it simple.
                    # If we need longer paths, we can relax this.
                    if len(path) - 1 >= found_dist:
                        continue
                    visited.add(nb)
                    path.append(nb)
                    dfs(nb, path, visited)
                    path.pop()
                    visited.remove(nb)
                    
            dfs(v, [v], {v})
            
            if not candidates:
                return False
                
            for path in candidates:
                interiors = set(path[1:-1])
                if interiors & used_new_interiors:
                    continue
                # Add to new_paths
                key = frozenset((v, u))
                new_paths[key] = path
                used_new_interiors.update(interiors)
                
                if backtrack(idx + 1):
                    return True
                    
                # Undo
                used_new_interiors.difference_update(interiors)
                del new_paths[key]
                
            return False
            
        if backtrack(0):
            return True, new_paths
        else:
            return False, None

    # Main loop: Try to expand each maximal clique
    for clique in max_cliques:
        if time_up(): break
        
        current_branches = list(clique)
        current_paths = {}
        # Initialize paths for the clique (edges)
        for u, v in combinations(current_branches, 2):
            current_paths[frozenset((u, v))] = [u, v]
            
        update_best(current_branches, list(current_paths.values()))
        
        # Try to expand
        changed = True
        while changed and not time_up():
            changed = False
            candidates = [v for v in range(n) if v not in current_branches]
            
            # Heuristic: Sort candidates by number of connections to current branches
            candidates.sort(key=lambda v: -len(adj[v] & set(current_branches)))
            
            for v in candidates:
                if time_up(): break
                success, new_paths_dict = try_add_vertex(v, current_branches, current_paths)
                if success:
                    # Update state
                    current_branches.append(v)
                    # Merge new paths
                    for key, path in new_paths_dict.items():
                        current_paths[key] = path
                    update_best(current_branches, list(current_paths.values()))
                    changed = True
                    break # Restart candidate loop with new set
                    
    # If we didn't find anything (should not happen), fallback
    if best_result is None:
        if edges:
            u, v = edges[0]
            best_result = {'branches': [u, v], 'paths': [[u, v]]}
        else:
            best_result = {'branches': [], 'paths': []}

    # Ensure types are correct
    best_result['branches'] = list(best_result['branches'])
    best_result['paths'] = [list(p) for p in best_result['paths']]
    
    return best_result

if __name__ == '__main__':
    res = construct()
    print(res)
