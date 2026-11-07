"""Solver for large clique subdivisions in a pinned graph."""

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def construct():
    import time
    from collections import deque
    from itertools import combinations

    t_start = time.time()
    DEADLINE = 82.0

    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    def time_up():
        return time.time() - t_start > DEADLINE

    def bfs_shortest(src, dst, forbidden):
        """BFS from src to dst avoiding forbidden interior vertices. Returns shortest path or None."""
        if src == dst:
            return [src]
        if src in forbidden and dst in forbidden:
            # both endpoints forbidden - only valid if path length 2
            if dst in adj[src]:
                return [src, dst]
            return None
        parent = {src: None}
        queue = deque([src])
        while queue:
            cur = queue.popleft()
            if cur == dst:
                path = []
                node = dst
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            for nb in sorted(adj[cur]):
                if nb not in parent:
                    if nb != dst and nb in forbidden:
                        continue
                    parent[nb] = cur
                    queue.append(nb)
        return None

    def bfs_random(src, dst, forbidden, rng, iters=8):
        """Try BFS with randomized neighbor ordering to find diverse paths."""
        if src == dst:
            return [src]
        for _ in range(iters):
            parent = {src: None}
            queue = deque([src])
            found = False
            while queue:
                cur = queue.popleft()
                if cur == dst:
                    found = True
                    break
                nbrs = list(adj[cur])
                rng.shuffle(nbrs)
                for nb in nbrs:
                    if nb not in parent:
                        if nb != dst and nb in forbidden:
                            continue
                        parent[nb] = cur
                        queue.append(nb)
            if found:
                path = []
                node = dst
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
        return None

    def find_path(src, dst, forbidden, rng):
        """Find a path from src to dst avoiding forbidden interiors.
        Try shortest first, then randomized for diversity."""
        p = bfs_shortest(src, dst, forbidden)
        if p is not None:
            return p
        return bfs_random(src, dst, forbidden, rng)

    def try_add_branch(branches, paths_dict, used_interiors, new_branch, rng):
        """Try to add new_branch to existing clique subdivision.
        Returns (new_paths_dict, new_used_interiors) or None."""
        bset = set(branches) | {new_branch}
        new_paths = dict(paths_dict)
        new_used = set(used_interiors)
        
        for b in branches:
            # Find path from new_branch to b
            path = find_path(new_branch, b, new_used, rng)
            if path is None:
                return None
            interior = set(path[1:-1])
            if interior & bset:
                return None
            if interior & new_used:
                return None
            new_used |= interior
            new_paths[(new_branch, b)] = path
            new_paths[(b, new_branch)] = path
        
        return new_paths, new_used

    def greedy_grow(seed_branches, rng, max_grow=12):
        """Greedy incremental growth: start from seed, add branches one by one."""
        branches = list(seed_branches)
        paths_dict = {}
        used_interiors = set()
        
        # Build initial paths for seed
        for i in range(len(branches)):
            for j in range(i+1, len(branches)):
                u, v = branches[i], branches[j]
                path = find_path(u, v, used_interiors, rng)
                if path is None:
                    return None
                interior = set(path[1:-1])
                bset = set(branches)
                if interior & bset:
                    return None
                if interior & used_interiors:
                    return None
                used_interiors |= interior
                paths_dict[(u, v)] = path
                paths_dict[(v, u)] = path
        
        bset = set(branches)
        
        for _ in range(max_grow):
            if time_up():
                break
            # Try to add each candidate not already in branches
            best_candidate = None
            best_result = None
            candidates = [v for v in range(n) if v not in bset]
            
            # Score candidates: prefer those with direct edges to many branches
            def cand_score(v):
                direct_edges = sum(1 for b in branches if b in adj[v])
                # Penalize if v is in used_interiors
                if v in used_interiors:
                    direct_edges -= 5
                return -direct_edges
            
            candidates.sort(key=cand_score)
            
            for cand in candidates:
                if time_up():
                    break
                result = try_add_branch(branches, paths_dict, used_interiors, cand, rng)
                if result is not None:
                    best_candidate = cand
                    best_result = result
                    break
            
            if best_candidate is None:
                break
            
            branches.append(best_candidate)
            paths_dict, used_interiors = best_result
            bset.add(best_candidate)
        
        if len(branches) < 2:
            return None
        
        path_list = []
        for i in range(len(branches)):
            for j in range(i+1, len(branches)):
                path_list.append(paths_dict[(branches[i], branches[j])])
        return {'branches': branches, 'paths': path_list}

    def backtrack_grow(branches, paths_dict, used_interiors, rng, depth=0):
        """Backtracking growth: try adding branches with backtracking."""
        if time_up():
            return None
        if depth > 14:
            return None
        
        bset = set(branches)
        candidates = [v for v in range(n) if v not in bset]
        
        def cand_score(v):
            direct_edges = sum(1 for b in branches if b in adj[v])
            if v in used_interiors:
                direct_edges -= 5
            return -direct_edges
        
        candidates.sort(key=cand_score)
        
        for cand in candidates[:10]:  # limit branching factor
            if time_up():
                return None
            result = try_add_branch(branches, paths_dict, used_interiors, cand, rng)
            if result is not None:
                new_paths, new_used = result
                # Recurse
                deeper = backtrack_grow(branches + [cand], new_paths, new_used, rng, depth+1)
                if deeper is not None:
                    return deeper
                # Also record current as a valid solution
                path_list = []
                for i in range(len(branches)):
                    for j in range(i+1, len(branches)):
                        path_list.append(paths_dict[(branches[i], branches[j])])
                return {'branches': list(branches), 'paths': path_list}
        
        # No branch could be added - return current
        if len(branches) >= 2:
            path_list = []
            for i in range(len(branches)):
                for j in range(i+1, len(branches)):
                    path_list.append(paths_dict[(branches[i], branches[j])])
            return {'branches': list(branches), 'paths': path_list}
        return None

    # Degree ranking
    deg = [len(adj[i]) for i in range(n)]
    candidates_sorted = sorted(range(n), key=lambda i: (-deg[i], i))

    best = None
    best_size = 0

    # Strategy 1: Greedy growth from various seeds
    import random
    rng = random.Random(42)
    
    # Try growing from different initial seeds
    # Seed with top-degree vertices
    for seed_size in [2, 3, 4]:
        if time_up():
            break
        top = candidates_sorted[:seed_size + 3]
        for seed_combo in combinations(top, seed_size):
            if time_up():
                break
            result = greedy_grow(list(seed_combo), rng)
            if result is not None and len(result['branches']) > best_size:
                best = result
                best_size = len(result['branches'])
                break  # found good seed, move on

    # Strategy 2: Backtracking growth from good seeds
    for seed_size in [2, 3]:
        if time_up():
            break
        if best_size >= 10:
            break
        top = candidates_sorted[:seed_size + 4]
        for seed_combo in combinations(top, seed_size):
            if time_up():
                break
            # First build initial paths
            branches = list(seed_combo)
            paths_dict = {}
            used_interiors = set()
            valid = True
            for i in range(len(branches)):
                for j in range(i+1, len(branches)):
                    u, v = branches[i], branches[j]
                    path = find_path(u, v, used_interiors, rng)
                    if path is None:
                        valid = False
                        break
                    interior = set(path[1:-1])
                    if interior & set(branches) or interior & used_interiors:
                        valid = False
                        break
                    used_interiors |= interior
                    paths_dict[(u, v)] = path
                    paths_dict[(v, u)] = path
                if not valid:
                    break
            if not valid:
                continue
            result = backtrack_grow(branches, paths_dict, used_interiors, rng)
            if result is not None and len(result['branches']) > best_size:
                best = result
                best_size = len(result['branches'])

    # Strategy 3: Try more diverse seeds with different RNG
    if best_size < 9 and not time_up():
        rng2 = random.Random(123)
        for seed_size in [2, 3]:
            if time_up():
                break
            for seed_combo in combinations(candidates_sorted[:8], seed_size):
                if time_up():
                    break
                result = greedy_grow(list(seed_combo), rng2)
                if result is not None and len(result['branches']) > best_size:
                    best = result
                    best_size = len(result['branches'])

    # Fallback: find a triangle
    if best is None:
        for a, b, c in combinations(range(n), 3):
            if b in adj[a] and c in adj[a] and c in adj[b]:
                best = {
                    'branches': [a, b, c],
                    'paths': [[a, b], [a, c], [b, c]]
                }
                break
        if best is None:
            u, v = edges[0]
            best = {'branches': [u, v], 'paths': [[u, v]]}

    # Convert to proper types
    branches = [int(x) for x in best['branches']]
    paths = [[int(x) for x in p] for p in best['paths']]
    return {'branches': branches, 'paths': paths}
