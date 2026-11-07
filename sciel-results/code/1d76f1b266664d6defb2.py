"""Solver for large clique subdivisions in a pinned graph."""

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def construct():
    import time
    from collections import deque
    from itertools import combinations
    import random

    t_start = time.time()
    DEADLINE = 85.0

    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    # Precompute sorted adjacency for deterministic traversal
    adj_sorted = [sorted(adj[i]) for i in range(n)]

    def time_up():
        return time.time() - t_start > DEADLINE

    def validate(branches, paths):
        """Validate a clique subdivision. Returns True if valid."""
        bset = set(branches)
        if len(branches) < 2:
            return False
        if len(branches) != len(bset):
            return False
        expected_paths = len(branches) * (len(branches) - 1) // 2
        if len(paths) != expected_paths:
            return False
        # Check each path
        used_interiors = set()
        for p in paths:
            if not isinstance(p, (list, tuple)):
                return False
            if len(p) < 2:
                return False
            u, v = p[0], p[-1]
            if u not in bset or v not in bset:
                return False
            if u == v:
                return False
            # Check endpoints are distinct branches
            # Check path is simple
            seen = set()
            for node in p:
                if node in seen:
                    return False
                seen.add(node)
            # Check interiors avoid branches and are disjoint
            for node in p[1:-1]:
                if node in bset:
                    return False
                if node in used_interiors:
                    return False
            used_interiors.update(p[1:-1])
            # Check adjacency
            for i in range(len(p) - 1):
                if p[i+1] not in adj[p[i]]:
                    return False
        # Check all pairs are covered
        path_pairs = set()
        for p in paths:
            path_pairs.add(frozenset((p[0], p[-1])))
        for i in range(len(branches)):
            for j in range(i + 1, len(branches)):
                if frozenset((branches[i], branches[j])) not in path_pairs:
                    return False
        return True

    def find_paths_for_branches(branches, rng, max_iters=200):
        """Given a set of branch vertices, find internally-disjoint paths between all pairs.
        Uses randomized BFS/DFS with retries. Returns dict of paths or None."""
        k = len(branches)
        if k < 2:
            return None
        
        # We need to find paths for all C(k,2) pairs
        # Strategy: repeatedly try to find paths, using randomized ordering
        # Use a "path assignment" approach: try to assign paths greedily
        
        for attempt in range(max_iters):
            if time_up():
                return None
            
            paths_dict = {}
            used_interiors = set()
            success = True
            
            # Order pairs: start with edges (direct connections) first
            pair_list = []
            for i in range(k):
                for j in range(i + 1, k):
                    pair_list.append((branches[i], branches[j]))
            
            # Sort: prefer direct edges, then shorter potential paths
            def pair_key(pair):
                u, v = pair
                if v in adj[u]:
                    return 0
                return 1
            
            pair_list.sort(key=pair_key)
            
            # Shuffle non-edge pairs for diversity
            edge_pairs = [p for p in pair_list if p[1] in adj[p[0]]]
            non_edge_pairs = [p for p in pair_list if p[1] not in adj[p[0]]]
            rng.shuffle(non_edge_pairs)
            pair_list = edge_pairs + non_edge_pairs
            
            for (u, v) in pair_list:
                if time_up():
                    return None
                # Find path from u to v avoiding used_interiors and other branches
                forbidden = set(used_interiors)
                forbidden.update(branches)
                forbidden.discard(u)
                forbidden.discard(v)
                
                path = _find_path_random(u, v, forbidden, rng, adj_sorted)
                if path is None:
                    success = False
                    break
                
                interior = set(path[1:-1])
                if interior & set(branches):
                    success = False
                    break
                if interior & used_interiors:
                    success = False
                    break
                
                used_interiors |= interior
                paths_dict[(u, v)] = path
                paths_dict[(v, u)] = path
            
            if success:
                # Convert to list
                path_list = []
                for (u, v) in pair_list:
                    path_list.append(paths_dict[(u, v)])
                return path_list
        
        return None

    def _find_path_random(src, dst, forbidden, rng, adj_sorted, max_depth=12):
        """Find a path from src to dst avoiding forbidden vertices in interior.
        Uses randomized DFS with depth limit."""
        if src == dst:
            return [src]
        if dst in adj[src] and src not in forbidden and dst not in forbidden:
            return [src, dst]
        
        # Randomized DFS
        def dfs(cur, path, visited):
            if len(path) > max_depth + 2:
                return None
            if cur == dst:
                return list(path)
            nbrs = list(adj_sorted[cur])
            rng.shuffle(nbrs)
            for nb in nbrs:
                if nb in visited:
                    continue
                if nb != dst and nb in forbidden:
                    continue
                visited.add(nb)
                path.append(nb)
                result = dfs(nb, path, visited)
                if result is not None:
                    return result
                path.pop()
                visited.discard(nb)
            return None
        
        visited = {src}
        return dfs(src, [src], visited)

    def _find_path_bfs(src, dst, forbidden, adj_sorted):
        """Standard BFS shortest path avoiding forbidden interiors."""
        if src == dst:
            return [src]
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
            for nb in adj_sorted[cur]:
                if nb not in parent:
                    if nb != dst and nb in forbidden:
                        continue
                    parent[nb] = cur
                    queue.append(nb)
        return None

    def find_paths_hybrid(branches, rng, max_iters=500):
        """Find paths using hybrid approach: BFS for edges, randomized DFS for others.
        Multiple restarts with different orderings."""
        k = len(branches)
        if k < 2:
            return None
        
        for attempt in range(max_iters):
            if time_up():
                return None
            
            paths_dict = {}
            used_interiors = set()
            success = True
            
            pair_list = []
            for i in range(k):
                for j in range(i + 1, k):
                    pair_list.append((branches[i], branches[j]))
            
            # Order: direct edges first, then by some heuristic
            def pair_key(pair):
                u, v = pair
                if v in adj[u]:
                    return (0, 0)
                # Count common non-forbidden neighbors as proxy for shortness
                common = len(adj[u] & adj[v])
                return (1, -common)
            
            pair_list.sort(key=pair_key)
            
            # For non-edge pairs, shuffle to get diversity
            edge_pairs = [(u,v) for u,v in pair_list if v in adj[u]]
            non_edge_pairs = [(u,v) for u,v in pair_list if v not in adj[u]]
            rng.shuffle(non_edge_pairs)
            pair_list = edge_pairs + non_edge_pairs
            
            for (u, v) in pair_list:
                if time_up():
                    return None
                forbidden = set(used_interiors)
                forbidden.update(branches)
                forbidden.discard(u)
                forbidden.discard(v)
                
                path = None
                if v in adj[u]:
                    path = [u, v]
                else:
                    # Try BFS first
                    path = _find_path_bfs(u, v, forbidden, adj_sorted)
                    if path is None:
                        path = _find_path_random(u, v, forbidden, rng, adj_sorted)
                
                if path is None:
                    success = False
                    break
                
                interior = set(path[1:-1])
                if interior & set(branches):
                    success = False
                    break
                if interior & used_interiors:
                    success = False
                    break
                
                used_interiors |= interior
                paths_dict[(u, v)] = path
                paths_dict[(v, u)] = path
            
            if success:
                path_list = []
                for (u, v) in pair_list:
                    path_list.append(paths_dict[(u, v)])
                return path_list
        
        return None

    # Degree ranking
    deg = [len(adj[i]) for i in range(n)]
    candidates_sorted = sorted(range(n), key=lambda i: (-deg[i], i))
    
    best = None
    best_size = 0

    def try_branch_set(branches, rng, max_iters=300):
        """Try to find valid paths for a given branch set. Update best if successful."""
        nonlocal best, best_size
        if len(branches) < 2:
            return False
        path_list = find_paths_hybrid(branches, rng, max_iters=max_iters)
        if path_list is not None:
            if validate(branches, path_list):
                if len(branches) > best_size:
                    best = {'branches': list(branches), 'paths': path_list}
                    best_size = len(branches)
                    return True
        return False

    # Strategy 1: Try all subsets of top-degree vertices
    # Start with smaller subsets and grow
    rng = random.Random(42)
    
    # First, find a good base by trying small cliques/subsets
    # Try all pairs of top vertices
    top_candidates = candidates_sorted[:12]
    
    # Try growing from pairs
    for i in range(len(top_candidates)):
        if time_up():
            break
        for j in range(i + 1, len(top_candidates)):
            if time_up():
                break
            seed = [top_candidates[i], top_candidates[j]]
            # Try to grow this seed
            current = list(seed)
            try_branch_set(current, rng, max_iters=100)
            
            # Greedy growth
            for _ in range(15):
                if time_up():
                    break
                added = False
                candidates = [v for v in top_candidates if v not in current]
                # Also include other vertices
                candidates += [v for v in range(n) if v not in current and v not in top_candidates]
                rng.shuffle(candidates)
                
                for cand in candidates:
                    if time_up():
                        break
                    new_set = current + [cand]
                    if try_branch_set(new_set, rng, max_iters=150):
                        current = new_set
                        added = True
                        break
                if not added:
                    break
    
    # Strategy 2: Systematic search over subsets of size k
    # For k from 3 to 10, try random subsets
    for k in range(3, 11):
        if time_up():
            break
        if best_size >= k:
            continue
        # Try random subsets of size k from top candidates
        for _ in range(50):
            if time_up():
                break
            subset = rng.sample(top_candidates, min(k, len(top_candidates)))
            try_branch_set(subset, rng, max_iters=200)
    
    # Strategy 3: Local search / hill climbing
    # Start from best found, try to add/remove/swap vertices
    if best is not None and not time_up():
        current_branches = list(best['branches'])
        current_paths = best['paths']
        
        for _ in range(200):
            if time_up():
                break
            improved = False
            # Try adding a vertex
            candidates = [v for v in range(n) if v not in current_branches]
            rng.shuffle(candidates)
            for cand in candidates[:20]:
                if time_up():
                    break
                new_set = current_branches + [cand]
                new_paths = find_paths_hybrid(new_set, rng, max_iters=200)
                if new_paths is not None and validate(new_set, new_paths):
                    current_branches = new_set
                    current_paths = new_paths
                    best = {'branches': list(current_branches), 'paths': current_paths}
                    best_size = len(current_branches)
                    improved = True
                    break
            if not improved:
                # Try swapping: remove one, add another
                if len(current_branches) > 2:
                    for remove_idx in range(len(current_branches)):
                        if time_up():
                            break
                        for cand in candidates[:10]:
                            if time_up():
                                break
                            new_set = current_branches[:remove_idx] + current_branches[remove_idx+1:] + [cand]
                            new_paths = find_paths_hybrid(new_set, rng, max_iters=200)
                            if new_paths is not None and validate(new_set, new_paths):
                                current_branches = new_set
                                current_paths = new_paths
                                best = {'branches': list(current_branches), 'paths': current_paths}
                                best_size = len(current_branches)
                                improved = True
                                break
                        if improved:
                            break
            if not improved:
                break
    
    # Fallback: find a triangle or edge
    if best is None:
        for a, b, c in combinations(range(n), 3):
            if b in adj[a] and c in adj[a] and c in adj[b]:
                best = {
                    'branches': [a, b, c],
                    'paths': [[a, b], [a, c], [b, c]]
                }
                break
        if best is None:
            for u, v in edges:
                best = {'branches': [u, v], 'paths': [[u, v]]}
                break
        if best is None:
            best = {'branches': [0, 1], 'paths': [[0, 1]]}

    # Final validation
    if not validate(best['branches'], best['paths']):
        # Emergency fallback
        for u, v in edges:
            if validate([u, v], [[u, v]]):
                best = {'branches': [u, v], 'paths': [[u, v]]}
                break

    # Convert to proper types
    branches = [int(x) for x in best['branches']]
    paths = [[int(x) for x in p] for p in best['paths']]
    return {'branches': branches, 'paths': paths}
