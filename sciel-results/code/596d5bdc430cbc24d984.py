import time
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

DEADLINE = time.time() + 85.0

def construct():
    t0 = time.time()
    n = P['n']
    edges = P['edges']
    
    # Build adjacency structure
    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
        
    def time_up():
        return time.time() - t0 > DEADLINE - 2.0

    def fallback():
        # Return a valid K2 subdivision
        u, v = edges[0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

    # Degrees
    degrees = [len(adj[i]) for i in range(n)]
    # Sort vertices by degree descending
    sorted_vertices = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    # We know K7 is possible. Let's try to find K8, K9, etc.
    # Candidate pool: top 15 vertices by degree
    pool_size = min(15, n)
    candidates = sorted_vertices[:pool_size]
    
    best = None
    best_size = 2
    
    # Initialize with a known valid solution if we can find one quickly
    # Or just start searching for larger ones
    
    # Function to check if a set of branches has a K_k subdivision
    def has_subdivision(branches):
        """
        Returns list of paths if valid, else None.
        branches: list of vertex ids
        """
        bs = set(branches)
        sorted_bs = sorted(branches)
        pairs = list(combinations(sorted_bs, 2))
        
        if not pairs:
            return []
        
        # Heuristic: Sort pairs by "tightness"
        # Tightness: number of neighbors of u and v that are NOT in bs
        # Lower tightness = more constrained
        def get_tightness(u, v):
            common = (adj[u] & adj[v]) - bs
            return len(common)
        
        # We try a few orderings
        orderings = [
            sorted(pairs, key=lambda p: get_tightness(p[0], p[1])),
            sorted(pairs, key=lambda p: -get_tightness(p[0], p[1])),
            pairs
        ]
        
        # Add random orderings
        rng = random.Random(123)
        for _ in range(2):
            shuffled = list(pairs)
            rng.shuffle(shuffled)
            orderings.append(shuffled)
        
        for ordering in orderings:
            if time_up():
                break
            sol = solve_paths(branches, bs, ordering)
            if sol is not None:
                # Reorder to standard combination order
                std_pairs = list(combinations(sorted_bs, 2))
                path_dict = {}
                for i, pair in enumerate(ordering):
                    path_dict[pair] = sol[i]
                
                result = []
                for p in std_pairs:
                    if p in path_dict:
                        result.append(path_dict[p])
                    else:
                        return None
                return result
        
        return None

    def solve_paths(branches, bs, ordered_pairs):
        """
        Backtracking solver.
        """
        num_pairs = len(ordered_pairs)
        chosen = [None] * num_pairs
        
        def find_paths(s, t, blocked, max_paths=10, max_len=10):
            """
            Find up to max_paths simple paths from s to t avoiding blocked (except s,t).
            blocked is a set of vertices.
            """
            paths = []
            
            # BFS for shortest path
            parent = {s: None}
            q = deque([s])
            found = False
            while q and not found:
                curr = q.popleft()
                for nb in adj[curr]:
                    if nb in blocked and nb != s and nb != t:
                        continue
                    if nb in parent:
                        continue
                    parent[nb] = curr
                    if nb == t:
                        found = True
                        break
                    q.append(nb)
            
            if found:
                path = []
                node = t
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                paths.append(path)
            
            if len(paths) >= max_paths:
                return paths
            
            # DFS for more paths
            def dfs(curr, path, visited):
                if len(paths) >= max_paths:
                    return
                if len(path) > max_len:
                    return
                if curr == t:
                    if path not in paths:
                        paths.append(list(path))
                    return
                for nb in adj[curr]:
                    if nb in visited:
                        continue
                    if nb in blocked and nb != s and nb != t:
                        continue
                    visited.add(nb)
                    path.append(nb)
                    dfs(nb, path, visited)
                    path.pop()
                    visited.remove(nb)
                    if len(paths) >= max_paths:
                        return
            
            dfs(s, [s], {s})
            return paths

        def solve(idx, used_internal):
            if idx == num_pairs:
                return True
            
            u, v = ordered_pairs[idx]
            blocked = used_internal | bs
            
            candidates = find_paths(u, v, blocked, max_paths=10, max_len=10)
            
            if not candidates:
                return False
            
            # Sort by length
            candidates.sort(key=len)
            
            for path in candidates:
                internal = set(path[1:-1])
                if internal & used_internal:
                    continue
                if internal & bs:
                    continue
                
                chosen[idx] = path
                for v_node in internal:
                    used_internal.add(v_node)
                
                if solve(idx + 1, used_internal):
                    return True
                
                for v_node in internal:
                    used_internal.discard(v_node)
                chosen[idx] = None
            
            return False
        
        if solve(0, set()):
            return chosen
        return None

    # Try to find larger subdivisions
    # Start from size 8 up to 10
    # We will iterate over combinations of the candidate pool
    
    # First, try to find K8
    # We'll try combinations of size 8 from the top 12 vertices
    # Then size 9 from top 14, etc.
    
    search_configs = [
        (8, 12),
        (9, 14),
        (10, 15)
    ]
    
    for k, pool_sz in search_configs:
        if time_up():
            break
        
        pool = sorted_vertices[:pool_sz]
        if len(pool) < k:
            continue
            
        # To avoid duplicates and redundant checks, we can just iterate
        # But we should prioritize "clique-like" sets
        # Let's just iterate all combinations for small k
        # C(12, 8) = 495
        # C(14, 9) = 2002
        # C(15, 10) = 3003
        
        # We'll break early if we find a solution
        found = False
        for comb in combinations(pool, k):
            if time_up():
                break
                
            # Optional: Prune if not connected enough?
            # Check if the subgraph induced by comb is connected? 
            # Actually, branches don't need to be connected in the subgraph, 
            # but the paths must exist.
            
            paths = has_subdivision(list(comb))
            if paths is not None:
                best = {
                    'branches': [int(x) for x in sorted(comb)],
                    'paths': [[int(x) for x in p] for p in paths]
                }
                best_size = k
                found = True
                # We can continue to see if we can find k+1?
                # The loop structure handles k+1 in the next iteration of search_configs
                # But we should break this inner loop to save time?
                # No, we want the BEST (largest) k.
                # So if we find a k=8, we record it, but we should still try to find k=9.
                # However, finding k=8 might be expensive if many fail.
                # Let's break the inner loop once we find a valid one for this k?
                # Yes, because any valid k is better than the previous best if k > best_size.
                # And we are iterating k in increasing order.
                # So if we find a valid k=8, we update best.
                # Then we move to k=9.
                # If we find a valid k=9, we update best.
                # So we only need ONE valid solution per k to update the best for that k.
                # But maybe a different set of k vertices is "better"? No, score is just count.
                # So we can break the inner loop.
                break
        
        if not found:
            # If we didn't find any for this k, maybe we should try a larger pool?
            # Or just proceed to next k?
            # If k=8 fails, k=9 is unlikely to succeed if the graph is sparse.
            # But let's try.
            pass

    # If we didn't find anything better than 2, we should try to find 3, 4, 5, 6, 7
    # The current program found 7.
    # Let's add a fallback search for smaller k if we failed to find 8+
    if best_size < 7:
        # Try to find 7
        pool = sorted_vertices[:15]
        for comb in combinations(pool, 7):
            if time_up():
                break
            paths = has_subdivision(list(comb))
            if paths is not None:
                best = {
                    'branches': [int(x) for x in sorted(comb)],
                    'paths': [[int(x) for x in p] for p in paths]
                }
                best_size = 7
                break
        
    if best is None:
        return fallback()
    
    return best

