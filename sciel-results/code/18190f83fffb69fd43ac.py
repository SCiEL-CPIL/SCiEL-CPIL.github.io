import time
from itertools import combinations
from collections import deque
import random

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = 85.0 # Safety margin for 90s limit
    
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

    def find_shortest_paths_bfs(u, v, banned, limit=50):
        """
        Find up to 'limit' shortest simple paths from u to v avoiding 'banned' vertices in interior.
        Returns list of paths sorted by length.
        """
        if u == v:
            return [[u]]
        
        # BFS to find shortest distance
        dist = {u: 0}
        parent = {u: None}
        q = deque([u])
        found_v = False
        while q:
            cur = q.popleft()
            if cur == v:
                found_v = True
                break
            for nb in adj[cur]:
                if nb in banned and nb != v:
                    continue
                if nb not in dist:
                    dist[nb] = dist[cur] + 1
                    parent[nb] = cur
                    q.append(nb)
        
        if not found_v or v not in dist:
            return []
        
        shortest_len = dist[v]
        
        # Now DFS to find all paths of length shortest_len
        # We use a visited set for the current path
        paths = []
        
        def dfs(cur, path, visited):
            if len(paths) >= limit:
                return
            if cur == v:
                if len(path) - 1 == shortest_len:
                    paths.append(list(path))
                return
            
            # Pruning: if current path length + min possible remaining > shortest_len, prune
            # But we don't have min possible remaining easily. 
            # However, since we are looking for shortest paths, we can just restrict depth.
            
            for nb in adj[cur]:
                if nb in visited:
                    continue
                if nb in banned and nb != v:
                    continue
                # Optimization: if we are at depth d, and dist[nb] is not defined or dist[nb] < dist[cur]+1?
                # Actually, BFS dist is from u. 
                # If we follow a path, the length is increasing.
                # We just need to ensure we don't exceed shortest_len.
                if len(path) - 1 >= shortest_len:
                    continue
                
                visited.add(nb)
                path.append(nb)
                dfs(nb, path, visited)
                path.pop()
                visited.remove(nb)
                
        dfs(u, [u], {u})
        return paths

    def find_disjoint_paths(branches):
        """
        Given a list of branches, find disjoint paths for all pairs.
        Returns list of paths if successful, else None.
        """
        if len(branches) < 2:
            return None
            
        pairs = list(combinations(branches, 2))
        
        # Heuristic: Sort pairs by shortest path length in the original graph (ignoring bans for now)
        # This helps to constrain the search space early.
        def pair_key(p):
            u, v = p
            # Quick BFS for length
            if u == v: return 0
            dist = {u: 0}
            q = deque([u])
            while q:
                cur = q.popleft()
                if cur == v:
                    return dist[cur]
                for nb in adj[cur]:
                    if nb not in dist:
                        dist[nb] = dist[cur] + 1
                        q.append(nb)
            return 1000
            
        pairs.sort(key=pair_key)
        
        paths = {}
        used_interiors = set()
        
        def backtrack(idx):
            if time_up():
                return False
            if idx == len(pairs):
                return True
                
            u, v = pairs[idx]
            # Find candidate paths avoiding used_interiors and other branches
            banned = set(branches) - {u, v}
            banned.update(used_interiors)
            
            # We need paths from u to v where interiors are not in banned
            candidates = find_shortest_paths_bfs(u, v, banned, limit=20)
            
            for path in candidates:
                # Check if path interiors are disjoint from used_interiors
                # (This is implicitly handled by 'banned' in find_shortest_paths_bfs? 
                #  Wait, find_shortest_paths_bfs takes 'banned' which includes used_interiors.
                #  So the returned paths should already avoid used_interiors.)
                
                # Update used_interiors
                new_interiors = set(path[1:-1])
                
                # Recursive step
                paths[(u, v)] = path
                old_used = set(used_interiors)
                used_interiors.update(new_interiors)
                
                if backtrack(idx + 1):
                    return True
                
                # Backtrack
                used_interiors.clear()
                used_interiors.update(old_used)
                del paths[(u, v)]
                
            return False
            
        if backtrack(0):
            # Return paths in a consistent order (list of lists)
            res_paths = []
            for p in pairs:
                res_paths.append(paths[p])
            return res_paths
        return None

    def verify_solution(branches, paths):
        """Verify a candidate solution. Returns True if valid."""
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

    # --- Search Strategy ---
    
    best_result = None
    best_size = 0
    
    def update_best(branches, paths):
        nonlocal best_result, best_size
        if verify_solution(branches, paths):
            if len(branches) > best_size:
                best_size = len(branches)
                best_result = {'branches': list(branches), 'paths': [list(p) for p in paths]}
                
    # 1. Find all cliques up to size 5
    cliques = []
    for size in range(3, 6):
        if time_up(): break
        for combo in combinations(range(n), size):
            if all((u, v) in edge_set for u, v in combinations(combo, 2)):
                cliques.append(list(combo))
                if time_up(): break
        if time_up(): break

    # Also add some random edges as starting points for K2 (though K2 is trivial, expanding from K2 is hard)
    # Better to start from K3+.
    
    # 2. Try to expand each clique
    # Sort cliques by size descending, then by density?
    cliques.sort(key=len, reverse=True)
    
    for clique in cliques:
        if time_up(): break
        
        current_branches = list(clique)
        # Verify initial clique
        initial_paths = [[u, v] for u, v in combinations(current_branches, 2)]
        # For a clique, the paths are just edges.
        # Check if edges exist (they do by definition of clique)
        # Check disjointness: interiors are empty, so always disjoint.
        update_best(current_branches, initial_paths)
        
        # Try to add vertices
        changed = True
        while changed and not time_up():
            changed = False
            candidates = [v for v in range(n) if v not in current_branches]
            
            # Heuristic: Sort candidates by number of connections to current branches
            candidates.sort(key=lambda v: -len(adj[v] & set(current_branches)))
            
            for v in candidates:
                if time_up(): break
                new_branches = current_branches + [v]
                paths = find_disjoint_paths(new_branches)
                if paths is not None:
                    current_branches = new_branches
                    update_best(current_branches, paths)
                    changed = True
                    break # Restart loop with new set
                    
    # 3. Local Search / Hill Climbing from best found so far
    if best_result is not None:
        current_branches = list(best_result['branches'])
        current_paths = [list(p) for p in best_result['paths']]
        
        # Try to swap vertices
        # If we are stuck, maybe a different set of same size is better for expansion?
        # Or try to remove a vertex and add two?
        
        # Let's try a random restart with a slightly different initial set
        # Take best branches, remove one, add a random neighbor?
        
        attempts = 0
        while not time_up() and attempts < 50:
            attempts += 1
            # Perturb
            perturbed = list(current_branches)
            if len(perturbed) > 2:
                # Remove a random vertex
                rem_idx = random.randint(0, len(perturbed)-1)
                perturbed.pop(rem_idx)
                # Add a random vertex not in perturbed
                candidates = [v for v in range(n) if v not in perturbed]
                if candidates:
                    add_v = random.choice(candidates)
                    perturbed.append(add_v)
            
            paths = find_disjoint_paths(perturbed)
            if paths is not None:
                update_best(perturbed, paths)
                if len(perturbed) > len(current_branches):
                    current_branches = perturbed
                    current_paths = paths

    # 4. Random Initialization
    # Try random sets of vertices of size 4, 5, 6
    for size in [4, 5, 6, 7]:
        if time_up(): break
        for _ in range(20):
            if time_up(): break
            random_set = random.sample(range(n), size)
            paths = find_disjoint_paths(random_set)
            if paths is not None:
                update_best(random_set, paths)
                # Expand from here
                current_branches = list(random_set)
                changed = True
                while changed and not time_up():
                    changed = False
                    candidates = [v for v in range(n) if v not in current_branches]
                    candidates.sort(key=lambda v: -len(adj[v] & set(current_branches)))
                    for v in candidates:
                        if time_up(): break
                        new_branches = current_branches + [v]
                        p = find_disjoint_paths(new_branches)
                        if p is not None:
                            current_branches = new_branches
                            update_best(current_branches, p)
                            changed = True
                            break

    if best_result is None:
        # Fallback: Find any edge
        if edges:
            u, v = edges[0]
            return {'branches': [u, v], 'paths': [[u, v]]}
        else:
            return {'branches': [], 'paths': []}
            
    return best_result

if __name__ == '__main__':
    res = construct()
    print(res)
