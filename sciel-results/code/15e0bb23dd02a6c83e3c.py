import time
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Build adjacency list
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Compute core numbers
    def compute_core_numbers(adj, n):
        degrees = [len(adj[i]) for i in range(n)]
        # Sort vertices by degree
        vertices = list(range(n))
        vertices.sort(key=lambda v: degrees[v])
        
        core = [0] * n
        i = 0
        while i < n:
            # Find max degree in remaining
            # Actually, standard algorithm:
            # Remove vertex with min degree, update neighbors
            
            # This simple implementation is O(N^2) which is fine for N=24
            # But let's use the bucket method or just a simple loop since N is small
            
            # Simple iterative removal
            # We need to be careful with the "remaining" set
            
            rem = set(range(n))
            deg = {v: len(adj[v] & rem) for v in rem}
            # This is getting complicated for a small N.
            # Let's just do a simple greedy approximation or exact for small N.
            
            # Exact Core Number for N=24 is fast enough with a simple O(N^2) or O(N^3) algo.
            # k-core: repeatedly remove vertices with degree < k.
            # Core number of v is the max k such that v is in the k-core.
            
            core_nums = [0] * n
            for k in range(1, n):
                rem = set(range(n))
                deg = {v: len(adj[v] & rem) for v in rem}
                changed = True
                while changed:
                    changed = False
                    to_remove = [v for v in rem if deg[v] < k]
                    if to_remove:
                        for v in to_remove:
                            rem.discard(v)
                        for v in to_remove:
                            for u in adj[v]:
                                if u in rem:
                                    deg[u] -= 1
                        changed = True
                # Vertices in rem have core number >= k
                for v in rem:
                    core_nums[v] = k
            return core_nums

    core_nums = compute_core_numbers(adj, n)
    
    best_branches = None
    best_paths = None
    
    # Helper: try to find k internally-disjoint paths between k branch vertices
    def try_branch_set(branches):
        """Try to find internally disjoint paths for all pairs in branches."""
        if len(branches) < 2:
            return None
        k = len(branches)
        branch_set = set(branches)
        pairs = list(combinations(branches, 2))
        
        used_interiors = set()
        
        def find_paths(remaining_pairs, used_int, current_paths):
            if time.time() > deadline:
                return None
            if not remaining_pairs:
                return current_paths
            
            # Heuristic: Sort pairs by shortest path length in current available graph
            # This helps to route "tight" paths first
            # We compute shortest path length for each pair avoiding used_int
            # If no path, return None immediately (pruning)
            
            scored_pairs = []
            for a, b in remaining_pairs:
                # Quick check: is there a path?
                # We can use a simple BFS to get length
                # But we do full BFS in the recursion step anyway.
                # Just sorting by static distance might be enough.
                scored_pairs.append((a, b))
                
            # To save time, we don't re-sort every time if it's expensive.
            # But N is small. Let's try sorting by the number of common neighbors in available graph?
            # Or just keep the order.
            
            for idx, (a, b) in enumerate(remaining_pairs):
                rest = remaining_pairs[:idx] + remaining_pairs[idx+1:]
                forbidden = used_int | (branch_set - {a, b})
                path = bfs_path(a, b, forbidden, adj)
                if path is not None:
                    interior = set(path[1:-1])
                    if interior & used_int:
                        continue
                    new_used = used_int | interior
                    result = find_paths(rest, new_used, current_paths + [path])
                    if result is not None:
                        return result
            return None
        
        import random
        rng = random.Random(42)
        
        result = find_paths(pairs, set(), [])
        if result is not None:
            return result
        
        for attempt in range(5):
            if time.time() > deadline:
                return None
            shuffled = pairs[:]
            rng.shuffle(shuffled)
            result = find_paths(shuffled, set(), [])
            if result is not None:
                return result
        
        return None
    
    def bfs_path(start, end, forbidden, adj):
        if start == end:
            return [start]
        visited = {start}
        parent = {start: None}
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if u == end:
                path = []
                node = end
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            for v in adj[u]:
                if v in visited:
                    continue
                if v in forbidden and v != end:
                    continue
                visited.add(v)
                parent[v] = u
                queue.append(v)
        return None
    
    # Greedy approach: start with a clique, then try to expand
    def find_max_clique_greedy(adj, n, core_nums):
        best = []
        # Start with high core number vertices
        starts = sorted(range(n), key=lambda v: core_nums[v], reverse=True)
        for start in starts:
            if time.time() > deadline:
                break
            clique = [start]
            candidates = set(adj[start])
            # Sort candidates by core number
            candidates = sorted(candidates, key=lambda v: core_nums[v], reverse=True)
            while candidates:
                # Pick vertex with most neighbors in clique, tie-break by core number
                # We need to check which candidates are actually adjacent to all in clique
                valid_candidates = [v for v in candidates if all(u in adj[v] for u in clique)]
                if not valid_candidates:
                    break
                best_v = max(valid_candidates, key=lambda v: (len(adj[v] & set(clique)), core_nums[v]))
                clique.append(best_v)
                # Update candidates
                new_candidates = []
                for v in valid_candidates:
                    if v != best_v:
                        new_candidates.append(v)
                candidates = new_candidates
            if len(clique) > len(best):
                best = clique[:]
        return best
    
    initial_clique = find_max_clique_greedy(adj, n, core_nums)
    
    paths_result = try_branch_set(initial_clique)
    if paths_result is not None:
        best_branches = initial_clique[:]
        best_paths = paths_result
    
    if best_branches is not None:
        current_branches = best_branches[:]
        expanded = True
        while expanded and time.time() < deadline:
            expanded = False
            current_set = set(current_branches)
            # Prune candidates: core number must be at least current size?
            # If we have K branches, the next one should be in the K-core?
            # Let's use core_nums[v] >= len(current_branches)
            min_core = len(current_branches)
            candidates = [v for v in range(n) if v not in current_set and core_nums[v] >= min_core]
            
            # If no candidates with high core, try all (but sorted by core)
            if not candidates:
                candidates = [v for v in range(n) if v not in current_set]
            
            candidates.sort(key=lambda v: core_nums[v], reverse=True)
            
            for v in candidates:
                if time.time() > deadline:
                    break
                new_branches = current_branches + [v]
                paths_result = try_branch_set(new_branches)
                if paths_result is not None:
                    current_branches = new_branches
                    best_branches = current_branches[:]
                    best_paths = paths_result
                    expanded = True
                    break
    
    import random
    rng2 = random.Random(123)
    
    for size in range(n, 1, -1):
        if time.time() > deadline:
            break
        if best_branches is not None and size <= len(best_branches):
            break
        for attempt in range(100):
            if time.time() > deadline:
                break
            # Sample from high core vertices preferentially?
            # Just random for now, but maybe weight by core?
            random_set = rng2.sample(range(n), size)
            paths_result = try_branch_set(random_set)
            if paths_result is not None:
                best_branches = random_set[:]
                best_paths = paths_result
                break
    
    if best_branches is None or len(best_branches) < 2:
        u, v = P['edges'][0]
        best_branches = [int(u), int(v)]
        best_paths = [[int(u), int(v)]]
    else:
        best_branches = [int(v) for v in best_branches]
        best_paths = [[int(v) for v in path] for path in best_paths]
    
    return {'branches': best_branches, 'paths': best_paths}
