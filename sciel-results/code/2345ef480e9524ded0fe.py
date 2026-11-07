import time
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin

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

    # Function to find all simple paths between u and v with length <= max_len
    # avoiding banned vertices (except u and v)
    def find_paths(u, v, banned, max_len=6):
        if u == v:
            return [[u]]
        paths = []
        
        # BFS to find shortest distance
        dist = {u: 0}
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
                    q.append(nb)
        
        if not found_v or v not in dist:
            return []
        
        shortest_len = dist[v]
        if shortest_len > max_len:
            return []

        # DFS to find all paths up to max_len
        def dfs(cur, path, visited):
            if len(paths) >= 20: # Limit number of paths to keep it fast
                return
            if cur == v:
                if len(path) - 1 <= max_len:
                    paths.append(list(path))
                return
            
            if len(path) - 1 >= max_len:
                return

            for nb in adj[cur]:
                if nb in visited:
                    continue
                if nb in banned and nb != v:
                    continue
                # Pruning: if current depth + min possible remaining > max_len
                # We don't have min remaining easily, but we can check if dist[nb] is known
                # and if dist[nb] + (len(path)-1) + 1 > max_len? No, dist is from u.
                # Just rely on max_len limit.
                
                visited.add(nb)
                path.append(nb)
                dfs(nb, path, visited)
                path.pop()
                visited.remove(nb)
                
        dfs(u, [u], {u})
        return paths

    def solve_disjoint_paths(branches):
        """
        Given a list of branches, find disjoint paths for all pairs.
        Returns list of paths if successful, else None.
        """
        if len(branches) < 2:
            return None
            
        pairs = list(combinations(branches, 2))
        
        # Heuristic: Sort pairs by the number of available paths in the initial graph
        # (ignoring other branch bans for a moment, or just using shortest path length)
        # Pairs with fewer options are more constraining.
        
        def pair_key(p):
            u, v = p
            # Count number of shortest paths? Or just shortest path length?
            # Shorter paths are often better because they use fewer vertices.
            # But pairs that are "far" might be harder to route disjointly.
            # Let's use the length of the shortest path in the full graph as a proxy.
            # Actually, we want to prioritize pairs that have FEWER options.
            # Let's just use shortest path length for now.
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
            # Banned vertices are all branches except u and v, plus used interiors
            banned = set(branches) - {u, v}
            banned.update(used_interiors)
            
            candidates = find_paths(u, v, banned, max_len=5) # Limit path length to 5
            
            for path in candidates:
                new_interiors = set(path[1:-1])
                
                # Check if new_interiors are disjoint from used_interiors
                # (This is handled by 'banned' in find_paths, but let's be safe)
                if new_interiors & used_interiors:
                    continue
                
                # Update state
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
            res_paths = []
            for p in pairs:
                res_paths.append(paths[p])
            return res_paths
        return None

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

    best_result = None
    best_size = 0

    def update_best(branches, paths):
        nonlocal best_result, best_size
        if verify_solution(branches, paths):
            if len(branches) > best_size:
                best_size = len(branches)
                best_result = {'branches': list(branches), 'paths': [list(p) for p in paths]}

    # 1. Find all maximal cliques
    # Simple Bron-Kerbosch or just check all combinations for small n
    # n=24, so checking all combinations of size 4 and 5 is feasible.
    cliques = []
    for size in [4, 5, 6]:
        if time_up(): break
        for combo in combinations(range(n), size):
            if all((u, v) in edge_set for u, v in combinations(combo, 2)):
                cliques.append(list(combo))
    cliques.sort(key=len, reverse=True)

    # 2. Try to expand each clique
    for clique in cliques:
        if time_up(): break
        current_branches = list(clique)
        initial_paths = [[u, v] for u, v in combinations(current_branches, 2)]
        update_best(current_branches, initial_paths)
        
        # Try to add vertices
        changed = True
        while changed and not time_up():
            changed = False
            candidates = [v for v in range(n) if v not in current_branches]
            # Sort candidates by connection to current branches
            candidates.sort(key=lambda v: -len(adj[v] & set(current_branches)))
            
            for v in candidates:
                if time_up(): break
                new_branches = current_branches + [v]
                paths = solve_disjoint_paths(new_branches)
                if paths is not None:
                    current_branches = new_branches
                    update_best(current_branches, paths)
                    changed = True
                    break # Restart loop with new set

    # 3. Local Search
    if best_result is not None:
        current_branches = list(best_result['branches'])
        
        for _ in range(100):
            if time_up(): break
            # Try to swap a vertex
            if len(current_branches) < 2: continue
            rem_idx = random.randint(0, len(current_branches)-1)
            rem_v = current_branches[rem_idx]
            new_branches = current_branches[:rem_idx] + current_branches[rem_idx+1:]
            
            candidates = [v for v in range(n) if v not in new_branches]
            candidates.sort(key=lambda v: -len(adj[v] & set(new_branches)))
            
            for v in candidates:
                if time_up(): break
                test_branches = new_branches + [v]
                paths = solve_disjoint_paths(test_branches)
                if paths is not None:
                    update_best(test_branches, paths)
                    if len(test_branches) > len(current_branches):
                        current_branches = test_branches
                    break # Found one, move to next iteration

    if best_result is None:
        if edges:
            u, v = edges[0]
            return {'branches': [u, v], 'paths': [[u, v]]}
        else:
            return {'branches': [], 'paths': []}
            
    return best_result

if __name__ == '__main__':
    res = construct()
    print(res)
