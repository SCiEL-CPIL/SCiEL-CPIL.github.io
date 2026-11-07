import time
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    t_start = time.time()
    DEADLINE = 85.0

    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    degrees = [len(adj[v]) for v in range(n)]
    sorted_verts = sorted(range(n), key=lambda x: degrees[x], reverse=True)

    def time_up():
        return time.time() - t_start > DEADLINE

    def find_shortest_path(u, v, banned):
        if u == v:
            return None
        if v in adj[u]:
            return [u, v]
        parent = {u: None}
        q = deque([u])
        while q:
            cur = q.popleft()
            for nxt in adj[cur]:
                if nxt == v:
                    path = [v]
                    node = cur
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path.reverse()
                    return path
                if nxt in parent or nxt in banned:
                    continue
                parent[nxt] = cur
                q.append(nxt)
        return None

    def find_all_paths_bfs(u, v, banned, max_len=8):
        """Find all simple paths from u to v with length <= max_len, respecting banned."""
        if u == v:
            return []
        if v in adj[u] and v not in banned:
            return [[u, v]]
        if u in banned and u != v:
            return []
        
        all_paths = []
        # DFS with depth limit
        def dfs(cur, path, depth):
            if len(all_paths) >= 50:
                return
            if depth > max_len:
                return
            for nxt in adj[cur]:
                if nxt == v:
                    all_paths.append(path + [v])
                    return
                if nxt in banned or nxt in path:
                    continue
                dfs(nxt, path + [nxt], depth + 1)
        
        dfs(u, [u], 1)
        return all_paths

    def try_find_subdivision(branches, rng, max_attempts=30):
        bset = set(branches)
        pairs = list(combinations(branches, 2))
        # Sort pairs by "difficulty": fewer available neighbors first
        def pair_difficulty(p):
            a, b = p
            avail_a = adj[a] - bset - {b}
            avail_b = adj[b] - bset - {a}
            return len(avail_a & avail_b)
        pairs.sort(key=pair_difficulty)
        
        for attempt in range(max_attempts):
            if time_up():
                break
            
            # Determine order
            if attempt == 0:
                order = list(pairs)
            elif attempt % 3 == 1:
                order = list(pairs)
                order.reverse()
            else:
                order = list(pairs)
                rng.shuffle(order)
            
            used_interior = set()
            paths = {}
            success = True
            
            for a, b in order:
                banned = used_interior | bset
                banned.discard(a)
                banned.discard(b)
                
                # Try shortest path first
                path = find_shortest_path(a, b, banned)
                
                if path is None:
                    # Try slightly longer paths
                    candidates = find_all_paths_bfs(a, b, banned, max_len=5)
                    found = False
                    for cand in candidates:
                        interior = set(cand[1:-1])
                        if not (interior & bset) and not (interior & used_interior):
                            path = cand
                            found = True
                            break
                    if not found:
                        success = False
                        break
                
                interior = set(path[1:-1])
                if interior & bset or interior & used_interior:
                    success = False
                    break
                
                used_interior |= interior
                paths[frozenset((a, b))] = path
            
            if success:
                return paths
        
        return None

    def try_find_subdivision_backtrack(branches, rng, max_nodes=30000):
        bset = set(branches)
        pairs = list(combinations(branches, 2))
        # Sort by constraint: pairs with fewer joint neighbors first
        pairs.sort(key=lambda p: len(adj[p[0]] & adj[p[1]]))
        
        used_interior = set()
        paths = {}
        nodes_visited = 0
        
        def bt(idx):
            nonlocal nodes_visited
            if time_up():
                return False
            if nodes_visited > max_nodes:
                return False
            if idx == len(pairs):
                return True
            
            a, b = pairs[idx]
            banned = used_interior | bset
            banned.discard(a)
            banned.discard(b)
            
            # Find candidate paths
            candidates = []
            
            if b in adj[a]:
                candidates.append([a, b])
            
            # BFS for shortest
            path = find_shortest_path(a, b, banned)
            if path:
                if path not in candidates:
                    candidates.append(path)
            
            # If we have few candidates, try to find more via DFS limited depth
            if len(candidates) < 5:
                all_paths = find_all_paths_bfs(a, b, banned, max_len=6)
                for p in all_paths:
                    if p not in candidates:
                        candidates.append(p)
            
            # Sort candidates by length (shorter is usually better for packing)
            candidates.sort(key=len)
            
            for path in candidates:
                interior = set(path[1:-1])
                if interior & bset or interior & used_interior:
                    continue
                
                nodes_visited += 1
                used_interior_new = used_interior | interior
                paths[frozenset((a, b))] = path
                
                if bt(idx + 1):
                    return True
                
                del paths[frozenset((a, b))]
            
            return False
        
        if bt(0):
            return paths
        return None

    def validate(branches, paths):
        if not branches or len(branches) < 2:
            return 0
        bset = set(branches)
        if len(bset) != len(branches):
            return 0
        expected_pairs = set(frozenset((a, b)) for a, b in combinations(branches, 2))
        if len(paths) != len(expected_pairs):
            return 0
        used_interior = set()
        for fkey, path in paths.items():
            if len(path) < 2:
                return 0
            a, b = path[0], path[-1]
            if a not in bset or b not in bset:
                return 0
            if a == b:
                return 0
            if fkey not in expected_pairs:
                return 0
            interior = set(path[1:-1])
            if interior & bset:
                return 0
            if interior & used_interior:
                return 0
            for i in range(len(path) - 1):
                if path[i + 1] not in adj[path[i]]:
                    return 0
            used_interior |= interior
        return len(branches)

    best_count = 0
    best_result = None
    rng = random.Random(42)

    # Phase 1: Top-down k search
    for k in range(10, 1, -1):
        if time_up():
            break
        
        candidates = set()
        
        # Top k by degree
        candidates.add(tuple(sorted_verts[:k]))
        
        # Top k-1 + random
        for _ in range(8):
            rest = [v for v in range(n) if v not in sorted_verts[:k-1]]
            if rest:
                cand = sorted_verts[:k-1] + [rng.choice(rest)]
                candidates.add(tuple(cand))
        
        # Random samples
        for _ in range(40):
            candidates.add(tuple(rng.sample(range(n), k)))
        
        # Local search: start with top k, try swapping
        base = sorted_verts[:k]
        for _ in range(15):
            if time_up():
                break
            cand = list(base)
            idx = rng.randint(0, k-1)
            rest = [v for v in range(n) if v not in cand]
            if rest:
                cand[idx] = rng.choice(rest)
                candidates.add(tuple(cand))
        
        for cand_tuple in candidates:
            if time_up():
                break
            cand = list(cand_tuple)
            
            # Try greedy first
            paths = try_find_subdivision(cand, rng, max_attempts=40)
            if paths is None:
                # Try backtracking
                paths = try_find_subdivision_backtrack(cand, rng, max_nodes=30000)
            
            if paths is not None:
                cnt = validate(cand, paths)
                if cnt > best_count:
                    best_count = cnt
                    best_result = (cand, paths)
                    break
        
        if best_result is not None and best_count >= k:
            break

    # Phase 2: Greedy incremental growth
    if not time_up():
        if best_result is not None:
            current = list(best_result[0])
        else:
            current = sorted_verts[:2]
        
        current_paths = try_find_subdivision(current, rng, max_attempts=50)
        if current_paths is None:
            current_paths = try_find_subdivision_backtrack(current, rng, max_nodes=30000)
        
        for _ in range(20):
            if time_up():
                break
            if current_paths is None:
                break
            
            added = False
            candidates_to_add = [v for v in range(n) if v not in set(current)]
            candidates_to_add.sort(key=lambda v: degrees[v], reverse=True)
            
            for v in candidates_to_add:
                if time_up():
                    break
                new_branches = current + [v]
                new_paths = try_find_subdivision(new_branches, rng, max_attempts=25)
                if new_paths is None:
                    new_paths = try_find_subdivision_backtrack(new_branches, rng, max_nodes=20000)
                
                if new_paths is not None:
                    cnt = validate(new_branches, new_paths)
                    if cnt > best_count:
                        best_count = cnt
                        best_result = (new_branches, new_paths)
                        current = new_branches
                        current_paths = new_paths
                        added = True
                        break
            if not added:
                break

    # Fallback
    if best_result is None:
        u, v = edges[0]
        branches = [u, v]
        paths = {frozenset((u, v)): [u, v]}
        best_result = (branches, paths)

    branches, paths = best_result

    branch_list = [int(b) for b in sorted(branches)]
    path_list = []
    for a, b in combinations(sorted(branches), 2):
        fkey = frozenset((a, b))
        p = paths.get(fkey)
        if p is None:
            p = [a, b]
        path_list.append([int(x) for x in p])

    return {'branches': branch_list, 'paths': path_list}
