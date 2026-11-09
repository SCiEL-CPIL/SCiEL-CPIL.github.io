import time
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']
EDGE_POS = {}
for i, (u, v) in enumerate(EDGE_ORDER):
    EDGE_POS[(min(u,v), max(u,v))] = i + 1  # 1-based

def edge_index(u, v):
    return EDGE_POS[(min(u,v), max(u,v))]

def find_hamilton_cycle(adj, n, time_limit=None, start_time=None):
    """Find a Hamilton cycle in graph with adjacency list adj.
    Returns list of vertices in cycle order, or None."""
    if time_limit is None:
        time_limit = 30
    if start_time is None:
        start_time = time.time()
    
    # Check if graph is connected enough
    # Each vertex must have degree >= 2
    for v in range(n):
        if len(adj[v]) < 2:
            return None
    
    # Use backtracking with Warnsdorff heuristic
    # Start from vertex 0, fix it to break symmetry
    visited = [False] * n
    path = [0]
    visited[0] = True
    
    def backtrack():
        if time.time() - start_time > time_limit:
            return None
        if len(path) == n:
            # Check if we can close the cycle
            if 0 in adj[path[-1]]:
                return path[:]
            return None
        
        cur = path[-1]
        # Get unvisited neighbors, sorted by degree (Warnsdorff)
        neighbors = [v for v in adj[cur] if not visited[v]]
        if not neighbors:
            return None
        
        # Sort by number of unvisited neighbors (prefer fewer)
        neighbors.sort(key=lambda v: len([x for x in adj[v] if not visited[x]]))
        
        for v in neighbors:
            visited[v] = True
            path.append(v)
            result = backtrack()
            if result is not None:
                return result
            path.pop()
            visited[v] = False
        
        return None
    
    return backtrack()

def find_hamilton_cycle_greedy(adj, n):
    """Try a greedy approach with multiple restarts."""
    import random
    rng = random.Random(42)
    
    best_cycle = None
    best_max_idx = float('inf')
    
    for attempt in range(100):
        # Random starting vertex and random ordering of neighbors
        start = rng.randint(0, n-1)
        visited = [False] * n
        path = [start]
        visited[start] = True
        
        success = True
        for _ in range(n - 1):
            cur = path[-1]
            neighbors = [v for v in adj[cur] if not visited[v]]
            if not neighbors:
                success = False
                break
            # Pick randomly or by heuristic
            v = rng.choice(neighbors)
            visited[v] = True
            path.append(v)
        
        if success and 0 in adj[path[-1]] or (start == 0 and start in adj[path[-1]]):
            # Check if it's a valid cycle
            if path[-1] in adj[start]:
                cycle = path[:]
                # Compute max edge index
                max_idx = 0
                for i in range(n):
                    u = cycle[i]
                    v = cycle[(i+1) % n]
                    idx = edge_index(u, v)
                    max_idx = max(max_idx, idx)
                if max_idx < best_max_idx:
                    best_max_idx = max_idx
                    best_cycle = cycle
    
    return best_cycle

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin
    
    # Binary search on T: smallest T such that edges 1..T contain a Hamilton cycle
    # Lower bound: 24 (need 24 edges), Upper bound: 276
    
    # First, let's try to find a good cycle using the full graph
    # Then binary search
    
    # Build adjacency for a given threshold T
    def build_adj(T):
        adj = [set() for _ in range(N)]
        for i in range(T):
            u, v = EDGE_ORDER[i]
            adj[u].add(v)
            adj[v].add(u)
        return [list(s) for s in adj]
    
    # Binary search
    lo, hi = 24, 276
    best_cycle = None
    best_T = 277
    
    # First check if full graph works
    adj_full = build_adj(276)
    cycle = find_hamilton_cycle(adj_full, N, time_limit=20, start_time=time.time())
    if cycle is not None:
        # Compute actual max index
        max_idx = 0
        for i in range(N):
            u = cycle[i]
            v = cycle[(i+1) % N]
            idx = edge_index(u, v)
            max_idx = max(max_idx, idx)
        best_T = max_idx
        best_cycle = cycle
    
    if time.time() - start_time > deadline:
        return best_cycle if best_cycle else list(range(N))
    
    # Binary search for smaller T
    while lo < hi and time.time() - start_time < deadline:
        mid = (lo + hi) // 2
        adj = build_adj(mid)
        
        # Quick check: all vertices must have degree >= 2
        valid = True
        for v in range(N):
            if len(adj[v]) < 2:
                valid = False
                break
        
        if valid:
            cycle = find_hamilton_cycle(adj, N, time_limit=10, start_time=time.time())
            if cycle is not None:
                # Verify
                max_idx = 0
                for i in range(N):
                    u = cycle[i]
                    v = cycle[(i+1) % N]
                    idx = edge_index(u, v)
                    max_idx = max(max_idx, idx)
                if max_idx <= mid:
                    hi = mid
                    best_T = max_idx
                    best_cycle = cycle
                else:
                    lo = mid + 1
            else:
                lo = mid + 1
        else:
            lo = mid + 1
    
    # Try to improve with local search if we have time
    if best_cycle is not None and time.time() - start_time < deadline - 5:
        # Try to find a cycle with lower max index using the found T as upper bound
        # Try multiple random restarts
        import random
        rng = random.Random(123)
        
        for T_try in range(best_T, 24, -1):
            if time.time() - start_time > deadline - 2:
                break
            adj = build_adj(T_try)
            valid = True
            for v in range(N):
                if len(adj[v]) < 2:
                    valid = False
                    break
            if not valid:
                continue
            
            # Try multiple times
            found = False
            for _ in range(20):
                if time.time() - start_time > deadline - 2:
                    break
                cycle = find_hamilton_cycle(adj, N, time_limit=3, start_time=time.time())
                if cycle is not None:
                    max_idx = 0
                    for i in range(N):
                        u = cycle[i]
                        v = cycle[(i+1) % N]
                        idx = edge_index(u, v)
                        max_idx = max(max_idx, idx)
                    if max_idx <= T_try:
                        best_T = max_idx
                        best_cycle = cycle
                        found = True
                        break
            if found:
                break
    
    if best_cycle is None:
        # Fallback: return identity permutation
        return list(range(N))
    
    return best_cycle

# EVOLVE-BLOCK-END
