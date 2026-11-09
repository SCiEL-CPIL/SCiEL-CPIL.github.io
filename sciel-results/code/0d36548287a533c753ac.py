import time
import sys
from functools import lru_cache

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

n = P['n']
edge_order = P['edge_order']

# Precompute edge indices (0-based)
edge_index = {}
for i, (u, v) in enumerate(edge_order):
    if u > v:
        u, v = v, u
    edge_index[(u, v)] = i

# Precompute edge weight = 1-based arrival index for each pair
INF = 277  # larger than any valid index
edge_weight = {}
for u in range(n):
    for v in range(u+1, n):
        edge_weight[(u, v)] = edge_index.get((u, v), INF) + 1

def get_edge_weight(u, v):
    if u > v:
        u, v = v, u
    return edge_weight[(u, v)]

def held_karp_bottleneck():
    """
    Find the minimum possible maximum edge weight (bottleneck) for a Hamiltonian cycle
    using DP over subsets. Returns (min_bottleneck, cycle) where cycle is a list of vertices.
    
    DP state: dp[S][v] = minimum bottleneck value for a path starting at vertex 0,
    visiting exactly the set S (which includes 0 and v), and ending at vertex v.
    
    We fix vertex 0 as the start to break rotational symmetry.
    """
    N = 1 << n
    
    # dp[S][v] = min bottleneck for path 0 -> ... -> v covering set S
    # Initialize with INF
    # Using a list of lists: dp[S] is a list of n values
    # Memory: 2^24 * 24 * 8 bytes ≈ 312 MB - might be too much
    # Let's use a more memory-efficient approach:
    # dp[S] = array of n values, but we can use a flat array or dict
    
    # Actually, let's use a list of arrays for memory efficiency
    # dp[S][v] stored as dp[S * n + v]
    
    # Total entries: 2^24 * 24 = 402,653,184 - this is ~3.2 GB with 8-byte ints. Too much.
    
    # Alternative: use a dict-based approach or level-by-level processing
    # Process by subset size: for each size k, process all subsets of size k
    # This reduces memory to O(C(n, k) * n) at any time
    
    # Let's use the level-by-level approach with dictionaries
    # dp[k][S][v] where S has exactly k bits set and v is in S
    # But even this is large.
    
    # Better approach: use numpy arrays for speed
    # dp is a 2D array: shape (N, n) where N = 2^n
    # N = 16,777,216, n = 24
    # Total: 402,653,184 floats (4 bytes) = ~1.6 GB - still too much
    
    # Let's think differently. The bottleneck DP can be done with binary search + feasibility check.
    # For a given threshold T, check if there's a Hamiltonian cycle using only edges with weight <= T.
    # This is just the original problem with a threshold.
    # But that's what we were already doing.
    
    # Actually, the Held-Karp for bottleneck can be done more cleverly.
    # Let's just do the binary search approach but with a much faster feasibility check.
    
    # Alternative: Since n=24, we can try a different approach.
    # Use the fact that we want to minimize the max edge weight.
    # We can use a variant of DP that tracks the bottleneck.
    
    # Let me reconsider: use numpy for the DP array
    import numpy as np
    
    # dp[S][v] = minimum bottleneck for path from 0 to v covering set S
    # We'll use a flat array of shape (N, n)
    # N = 2^24 = 16777216
    # This is 16777216 * 24 = 402653184 values
    # Using uint16 (max value 65535 > 277): 402653184 * 2 = ~805 MB
    # This might work if we have enough memory
    
    # Actually, let's use a smarter approach:
    # Process by subset size to keep memory bounded
    # For each subset size k from 1 to n:
    #   For each subset S of size k that contains 0:
    #     For each v in S (v != 0 if k > 1):
    #       dp[S][v] = min over u in S\{v}: max(dp[S\{v}][u], w(u,v))
    
    # But iterating over all subsets is still 2^24 operations total.
    # Let's just do it with numpy for speed.
    
    N = 1 << n
    # Use uint16 to save memory
    dp = np.full((N, n), 65535, dtype=np.uint16)
    
    # Base case: dp[{0}][0] = 0 (path of just vertex 0, no edges, bottleneck 0)
    dp[1][0] = 0  # subset {0} has bit 0 set, so mask = 1
    
    # Precompute: for each subset mask, list of vertices in the subset
    # This is expensive to precompute for all 2^24 subsets
    # Instead, iterate and compute on the fly
    
    # For each subset size from 1 to n:
    for size in range(1, n + 1):
        if time.time() > 80:  # safety margin
            break
        # Iterate over all subsets of size 'size' that contain vertex 0
        # We can iterate over all masks and check popcount, but that's slow
        # Better: iterate over combinations
        
        # For efficiency, let's iterate over all masks and check if they have the right size
        # and contain vertex 0
        # This is 2^24 * 24 operations per size level, which is too slow in Python
        
        # Let me reconsider the approach entirely.
        
        # Given the memory and time constraints, let's use a different strategy:
        # Binary search on the bottleneck value T, and for each T, check feasibility
        # using a fast backtracking with strong pruning.
        
        # Actually, let's just implement the binary search with an optimized backtracker.
        return None  # Fall through to the backtracking approach
    
    # This function is getting too complex. Let me simplify.
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin before 90s limit
    
    # Seed cycle: 0-1-2-...-23-0
    seed_cycle = list(range(n))
    max_idx = -1
    for i in range(n):
        u = seed_cycle[i]
        v = seed_cycle[(i+1) % n]
        if u > v:
            u, v = v, u
        idx = edge_index.get((u, v), -1)
        if idx > max_idx:
            max_idx = idx
    seed_score = max_idx + 1
    
    best_cycle = seed_cycle
    best_score = seed_score
    
    # Binary search for the minimum K (threshold)
    # lo: no Hamiltonian cycle with edges having index < lo
    # hi: Hamiltonian cycle exists with edges having index < hi
    lo = n  # minimum 24 edges needed
    hi = len(edge_order) + 1  # 277, all edges available
    
    # First, verify hi works (it should, since K_24 has Hamiltonian cycles)
    # We know the seed cycle works, so hi is valid
    
    while lo < hi:
        if time.time() > deadline:
            break
            
        mid = (lo + hi) // 2
        cycle = _find_hamiltonian_cycle_threshold(mid)
        
        if cycle is not None:
            # Found a cycle using only edges with 1-based index <= mid
            # Compute actual score
            actual_max = 0
            for i in range(n):
                u = cycle[i]
                v = cycle[(i+1) % n]
                if u > v:
                    u, v = v, u
                w = edge_weight[(u, v)]
                if w > actual_max:
                    actual_max = w
            actual_score = actual_max
            if actual_score < best_score:
                best_score = actual_score
                best_cycle = cycle
            hi = mid
        else:
            lo = mid + 1
    
    return best_cycle

def _find_hamiltonian_cycle_threshold(T):
    """
    Find a Hamiltonian cycle using only edges with 1-based arrival index <= T.
    Returns the cycle as a list of vertices, or None if no such cycle exists.
    Uses backtracking with strong pruning.
    """
    # Build adjacency list for edges with weight <= T
    adj = [[] for _ in range(n)]
    adj_set = [set() for _ in range(n)]
    for i in range(min(T, len(edge_order))):
        u, v = edge_order[i]
        adj[u].append(v)
        adj[v].append(u)
        adj_set[u].add(v)
        adj_set[v].add(u)
    
    # Quick check: all vertices must have degree >= 2
    for i in range(n):
        if len(adj[i]) < 2:
            return None
    
    # Sort neighbors by degree (ascending) for Warnsdorff's heuristic
    for i in range(n):
        adj[i].sort(key=lambda x: len(adj[x]))
    
    # Backtracking
    visited = [False] * n
    path = [0]
    visited[0] = True
    
    # Precompute: for each vertex, the set of its neighbors
    # Use bitmasks for fast operations
    n_mask = (1 << n) - 1
    
    def _dfs(u, depth):
        if depth == n:
            # Check if u is connected to 0
            if 0 in adj_set[u]:
                return True
            return False
        
        # Pruning: check if any unvisited vertex has all its neighbors visited
        # (except possibly u and the start)
        # This is expensive, so only do it at certain depths
        
        # Warnsdorff: try neighbors with fewest unvisited neighbors first
        neighbors = adj[u]
        # Sort by number of unvisited neighbors
        if len(neighbors) > 1:
            neighbors = sorted(neighbors, key=lambda v: (
                sum(1 for w in adj_set[v] if not visited[w])
            ))
        
        for v in neighbors:
            if not visited[v]:
                # Pruning: if v is the last unvisited vertex, check it connects to 0
                unvisited_count = n - depth - 1
                if unvisited_count == 0:
                    if 0 in adj_set[v]:
                        visited[v] = True
                        path.append(v)
                        return True
                    continue
                
                # Pruning: check if any unvisited vertex (other than v) would be isolated
                # after visiting v
                # This is expensive, skip for now
                
                visited[v] = True
                path.append(v)
                if _dfs(v, depth + 1):
                    return True
                path.pop()
                visited[v] = False
        
        return False
    
    if _dfs(0, 1):
        return path
    return None

# EVOLVE-BLOCK-END
