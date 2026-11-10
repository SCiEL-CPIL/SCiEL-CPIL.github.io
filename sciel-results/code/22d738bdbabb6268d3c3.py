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
    
    best_branches = None
    best_paths = None
    
    # Helper: try to find k internally-disjoint paths between k branch vertices
    # For a set of branches B, we need one simple path for each unordered pair
    # All path interiors must be pairwise disjoint and avoid all branches
    
    def try_branch_set(branches):
        """Try to find internally disjoint paths for all pairs in branches."""
        if len(branches) < 2:
            return None
        k = len(branches)
        branch_set = set(branches)
        pairs = list(combinations(branches, 2))
        
        # We need to route |pairs| paths through the graph
        # Each path goes from branch_i to branch_j (i < j)
        # Interiors must be disjoint and avoid all branches
        
        # Strategy: iteratively find shortest paths avoiding already-used interior vertices
        # Order pairs to maximize chance of success
        # Try multiple orderings if needed
        
        # First, sort pairs by "difficulty" - use shortest path length as proxy
        # Actually, let's just try all pairs in order and use BFS
        # To improve, try different orderings
        
        used_interiors = set()
        
        # Try to find paths for each pair
        # Use a greedy approach with backtracking on ordering
        def find_paths(remaining_pairs, used_int, current_paths):
            if time.time() > deadline:
                return None
            if not remaining_pairs:
                return current_paths
            
            # Try each pair first
            for idx, (a, b) in enumerate(remaining_pairs):
                rest = remaining_pairs[:idx] + remaining_pairs[idx+1:]
                # Find path from a to b avoiding used_int and all branches (except a, b)
                forbidden = used_int | (branch_set - {a, b})
                path = bfs_path(a, b, forbidden, adj)
                if path is not None:
                    # Interior vertices
                    interior = set(path[1:-1])
                    if interior & used_int:
                        continue
                    new_used = used_int | interior
                    result = find_paths(rest, new_used, current_paths + [path])
                    if result is not None:
                        return result
            return None
        
        # Start with a few different orderings
        import random
        rng = random.Random(42)
        
        # Try natural order first
        result = find_paths(pairs, set(), [])
        if result is not None:
            return result
        
        # Try a few randomized orderings
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
        """Find shortest path from start to end avoiding forbidden vertices (except start/end)."""
        if start == end:
            return [start]
        visited = {start}
        parent = {start: None}
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if u == end:
                # Reconstruct path
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
    
    # Greedy approach: start with a clique (fast), then try to expand
    # First, find all maximal cliques to get a good starting set
    
    # Actually, let's use a different strategy:
    # 1. Start with the seed clique
    # 2. Try to add vertices one by one
    # 3. Also try random sets of increasing size
    
    # Find a clique first (fast)
    def find_max_clique_greedy(adj, n):
        """Simple greedy: repeatedly pick vertex with most neighbors in current set."""
        # Try multiple starting vertices
        best = []
        for start in range(n):
            if time.time() > deadline:
                break
            clique = [start]
            candidates = set(adj[start])
            while candidates:
                # Pick vertex with most neighbors in clique
                best_v = max(candidates, key=lambda v: len(adj[v] & set(clique)))
                if len(adj[best_v] & set(clique)) < len(clique):
                    break
                clique.append(best_v)
                candidates = candidates & adj[best_v]
                candidates = candidates - set(clique)
            if len(clique) > len(best):
                best = clique[:]
        return best
    
    # Get initial clique
    initial_clique = find_max_clique_greedy(adj, n)
    
    # Try the clique
    paths_result = try_branch_set(initial_clique)
    if paths_result is not None:
        best_branches = initial_clique[:]
        best_paths = paths_result
    
    # Try to expand: add vertices one at a time
    if best_branches is not None:
        current_branches = best_branches[:]
        expanded = True
        while expanded and time.time() < deadline:
            expanded = False
            current_set = set(current_branches)
            # Try adding each non-branch vertex
            candidates = [v for v in range(n) if v not in current_set]
            # Sort by degree (heuristic)
            candidates.sort(key=lambda v: len(adj[v]), reverse=True)
            
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
    
    # Also try random sets of various sizes
    import random
    rng2 = random.Random(123)
    
    # Try sizes from large to small
    for size in range(n, 1, -1):
        if time.time() > deadline:
            break
        if best_branches is not None and size <= len(best_branches):
            break
        for attempt in range(100):
            if time.time() > deadline:
                break
            random_set = rng2.sample(range(n), size)
            paths_result = try_branch_set(random_set)
            if paths_result is not None:
                best_branches = random_set[:]
                best_paths = paths_result
                break
    
    # Ensure we have at least 2 branches
    if best_branches is None or len(best_branches) < 2:
        # Fallback: use an edge
        u, v = P['edges'][0]
        best_branches = [int(u), int(v)]
        best_paths = [[int(u), int(v)]]
    else:
        # Convert to proper Python types
        best_branches = [int(v) for v in best_branches]
        best_paths = [[int(v) for v in path] for path in best_paths]
    
    return {'branches': best_branches, 'paths': best_paths}
