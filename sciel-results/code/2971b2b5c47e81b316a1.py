"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    def bfs_path(src, dst, forbidden):
        """Find shortest path from src to dst avoiding forbidden vertices (except src/dst)."""
        if src == dst:
            return [src]
        visited = {src}
        queue = deque([(src, [src])])
        while queue:
            node, path = queue.popleft()
            for nb in sorted(adj[node]):
                if nb == dst:
                    return path + [dst]
                if nb not in forbidden and nb not in visited:
                    visited.add(nb)
                    queue.append((nb, path + [nb]))
        return None
    
    def find_k_shortest_paths(src, dst, forbidden, k=5):
        """Find up to k internally disjoint-ish paths from src to dst avoiding forbidden."""
        if src == dst:
            return [[src]]
        results = []
        visited = {src}
        queue = deque([(src, [src])])
        while queue and len(results) < k:
            node, path = queue.popleft()
            if len(path) > 12:
                continue
            for nb in sorted(adj[node]):
                if nb == dst:
                    results.append(path + [dst])
                    if len(results) >= k:
                        return results
                elif nb not in forbidden and nb not in visited:
                    visited.add(nb)
                    queue.append((nb, path + [nb]))
        return results

    def try_build_subdivision(branches, deadline, rng=None):
        """Try to find internally disjoint paths for all pairs in branches.
        Uses randomized backtracking. Returns list of paths or None."""
        if rng is None:
            rng = random
        branches = list(branches)
        branch_set = set(branches)
        pairs = list(itertools.combinations(branches, 2))
        
        # Score pairs: direct edges first, then by BFS distance
        pair_scores = []
        for (u, v) in pairs:
            if v in adj[u]:
                pair_scores.append((0, 0, u, v))
            else:
                # Quick BFS distance
                dist = 100
                if u != v:
                    seen = {u}
                    q = deque([(u, 0)])
                    while q:
                        node, d = q.popleft()
                        if d >= dist:
                            break
                        for nb in adj[node]:
                            if nb == v:
                                dist = d + 1
                                break
                            if nb not in seen and nb not in branch_set:
                                seen.add(nb)
                                q.append((nb, d + 1))
                pair_scores.append((1, dist, u, v))
        pair_scores.sort()
        ordered_pairs = [(u, v) for _, _, u, v in pair_scores]
        
        def backtrack(idx, used_internal, paths):
            if time.time() > deadline:
                return False
            if idx == len(ordered_pairs):
                return True
            u, v = ordered_pairs[idx]
            forbidden = (branch_set - {u, v}) | used_internal
            # Find multiple candidate paths
            candidates = find_k_shortest_paths(u, v, forbidden, k=4)
            if not candidates:
                return False
            # Shuffle for randomization
            rng.shuffle(candidates)
            for path in candidates:
                new_used = used_internal | set(path[1:-1])
                if backtrack(idx + 1, new_used, paths + [path]):
                    return True
            return False
        
        if backtrack(0, set(), []):
            return paths if False else None  # placeholder
        # Re-run to capture paths
        def backtrack2(idx, used_internal, paths):
            if time.time() > deadline:
                return False, None
            if idx == len(ordered_pairs):
                return True, paths
            u, v = ordered_pairs[idx]
            forbidden = (branch_set - {u, v}) | used_internal
            candidates = find_k_shortest_paths(u, v, forbidden, k=4)
            if not candidates:
                return False, None
            rng.shuffle(candidates)
            for path in candidates:
                new_used = used_internal | set(path[1:-1])
                ok, res = backtrack2(idx + 1, new_used, paths + [path])
                if ok:
                    return True, res
            return False, None
        
        ok, result = backtrack2(0, set(), [])
        if ok:
            return result
        return None
    
    def find_max_clique(deadline):
        """Find maximal cliques using Bron-Kerbosch."""
        cliques = []
        
        def bronk(R, P, X):
            if time.time() > deadline:
                return
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(list(R))
                return
            # Pivot selection
            if P or X:
                pivot = max(P | X, key=lambda v: len(adj[v] & P))
            else:
                return
            candidates = [v for v in P if v not in adj[pivot]]
            for v in candidates:
                bronk(R | {v}, P & adj[v], X & adj[v])
                P = P - {v}
                X = X | {v}
        
        bronk(set(), set(range(n)), set())
        cliques.sort(key=len, reverse=True)
        return cliques
    
    start_time = time.time()
    deadline = start_time + 75.0  # safety margin for 90s limit
    
    # Phase 1: Find cliques
    cliques = find_max_clique(deadline)
    best_result = None
    best_size = 0
    
    # Try each clique with randomized backtracking
    for clique in cliques:
        if time.time() > deadline:
            break
        if len(clique) < 2:
            break
        if len(clique) <= best_size:
            continue
        # Try multiple times with different random seeds
        for attempt in range(3):
            rng = random.Random(hash((tuple(clique), attempt)))
            result = try_build_subdivision(clique, deadline, rng)
            if result is not None:
                best_result = {'branches': list(clique), 'paths': result}
                best_size = len(clique)
                break
    
    # Phase 2: Greedy growth from best clique with backtracking
    if best_size >= 2:
        base_branches = list(best_result['branches'])
    else:
        # Fallback: find an edge
        u, v = P['edges'][0]
        base_branches = [u, v]
        best_size = 2
        best_result = {'branches': base_branches, 'paths': [[u, v]]}
    
    if time.time() < deadline:
        current_branches = list(base_branches)
        
        for iteration in range(80):
            if time.time() > deadline - 3:
                break
            
            branch_set = set(current_branches)
            candidates = [v for v in range(n) if v not in branch_set]
            random.shuffle(candidates)
            
            added = False
            for v in candidates:
                if time.time() > deadline - 3:
                    break
                new_branches = current_branches + [v]
                # Try with a few random seeds
                found = False
                for attempt in range(2):
                    rng = random.Random(hash((tuple(new_branches), attempt)))
                    result = try_build_subdivision(new_branches, deadline, rng)
                    if result is not None:
                        current_branches = new_branches
                        best_result = {'branches': list(current_branches), 'paths': result}
                        best_size = len(current_branches)
                        found = True
                        break
                if found:
                    added = True
                    break
            
            if not added:
                # Try swap: remove one, add one
                if len(current_branches) >= 3:
                    swap_done = False
                    for i in range(len(current_branches)):
                        if time.time() > deadline - 3:
                            break
                        temp = current_branches[:i] + current_branches[i+1:]
                        temp_set = set(temp)
                        candidates2 = [v for v in range(n) if v not in temp_set]
                        random.shuffle(candidates2)
                        for v in candidates2[:15]:
                            if time.time() > deadline - 3:
                                break
                            new_branches = temp + [v]
                            for attempt in range(2):
                                rng = random.Random(hash((tuple(new_branches), i, attempt)))
                                result = try_build_subdivision(new_branches, deadline, rng)
                                if result is not None:
                                    current_branches = new_branches
                                    best_result = {'branches': list(current_branches), 'paths': result}
                                    swap_done = True
                                    break
                            if swap_done:
                                break
                        if swap_done:
                            break
    
    # Phase 3: Random restarts from different clique subsets
    if time.time() < deadline - 5:
        for clique in cliques[:15]:
            if time.time() > deadline - 5:
                break
            if len(clique) < 2:
                continue
            rb = list(clique)
            random.shuffle(rb)
            
            for _ in range(40):
                if time.time() > deadline - 5:
                    break
                rb_set = set(rb)
                candidates = [v for v in range(n) if v not in rb_set]
                random.shuffle(candidates)
                added = False
                for v in candidates:
                    if time.time() > deadline - 5:
                        break
                    new_branches = rb + [v]
                    for attempt in range(2):
                        rng = random.Random(hash((tuple(new_branches), attempt)))
                        result = try_build_subdivision(new_branches, deadline, rng)
                        if result is not None:
                            rb = new_branches
                            if len(rb) > best_size:
                                best_size = len(rb)
                                best_result = {'branches': list(rb), 'paths': result}
                            added = True
                            break
                    if added:
                        break
                if not added:
                    break
    
    if best_result is not None:
        branches = [int(b) for b in best_result['branches']]
        paths = [[int(x) for x in p] for p in best_result['paths']]
        return {'branches': branches, 'paths': paths}
    
    # Final fallback
    u, v = P['edges'][0]
    return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
# EVOLVE-BLOCK-END
