"""Find a Hamilton cycle with minimal maximum edge arrival index in a fixed edge ordering."""
import time
import random
from itertools import combinations

# The pinned edge order (1-indexed arrival)
EDGE_ORDER = [
    [3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23],
    [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19],
    [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19],
    [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8],
    [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17],
    [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23],
    [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4],
    [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12],
    [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20],
    [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16],
    [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18],
    [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21],
    [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14],
    [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6],
    [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8],
    [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18],
    [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12],
    [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23],
    [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20],
    [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5],
    [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11],
    [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7],
    [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9],
    [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4],
    [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14],
    [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23],
    [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16],
    [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]
]

N = 24
TOTAL_EDGES = len(EDGE_ORDER)  # 276

# Build edge index: for each edge (u,v) with u<v, store its 1-based position
EDGE_INDEX = {}
for i, (u, v) in enumerate(EDGE_ORDER, 1):
    if u > v:
        u, v = v, u
    EDGE_INDEX[(u, v)] = i

def get_edge_pos(u, v):
    if u > v:
        u, v = v, u
    return EDGE_INDEX[(u, v)]

def cycle_max_index(cycle):
    """Compute the max edge arrival index for a Hamilton cycle (list of 24 vertices)."""
    mx = 0
    for i in range(N):
        u = cycle[i]
        v = cycle[(i + 1) % N]
        pos = get_edge_pos(u, v)
        if pos > mx:
            mx = pos
    return mx

def find_ham_cycle_in_prefix(K):
    """
    Try to find a Hamilton cycle using only edges with arrival index <= K.
    Uses DFS with bitmasks. Returns the cycle as a list or None.
    """
    # Build adjacency: adj[u] = bitmask of vertices connected to u with edge index <= K
    adj = [0] * N
    for (u, v), idx in EDGE_INDEX.items():
        if idx <= K:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    
    # Check degree: every vertex must have degree >= 2
    for v in range(N):
        if bin(adj[v]).count('1') < 2:
            return None
    
    # Fix vertex 0 as start to reduce symmetry
    # DFS with bitmask
    start = 0
    full_mask = (1 << N) - 1
    
    # Use iterative DFS to avoid recursion depth issues
    # State: (current_vertex, visited_mask, path)
    # For efficiency, we'll use a stack but this can blow up.
    # Instead, let's use a smarter approach: try to build the cycle.
    
    # Actually for N=24, even with pruning, DFS might be too slow in worst case.
    # Let's use a different strategy: randomized local search.
    return None

def randomized_search(K, max_iters=50000, rng=None):
    """
    Randomized local search to find a Hamilton cycle using only edges with index <= K.
    Returns cycle or None.
    """
    if rng is None:
        rng = random.Random()
    
    # Build adjacency list for edges with index <= K
    adj = [[] for _ in range(N)]
    for (u, v), idx in EDGE_INDEX.items():
        if idx <= K:
            adj[u].append(v)
            adj[v].append(u)
    
    # Check minimum degree
    for v in range(N):
        if len(adj[v]) < 2:
            return None
    
    # Try to find a Hamiltonian cycle using randomized DFS
    start = 0
    full_mask = (1 << N) - 1
    
    # Use a randomized approach: build a path and try to close it
    best_cycle = None
    
    for attempt in range(max_iters):
        # Randomized DFS
        visited = 1 << start
        path = [start]
        current = start
        
        # Shuffle adjacency lists for randomization
        adj_shuffled = [list(a) for a in adj]
        for a in adj_shuffled:
            rng.shuffle(a)
        
        success = False
        # Iterative DFS with backtracking
        stack = [(start, 1 << start, [start], 0)]
        
        # This is getting complex. Let's use a simpler heuristic: 
        # Try to build a Hamiltonian path from 0 to some vertex, then close the cycle.
        
        # Simpler: use a greedy randomized approach
        visited = [False] * N
        visited[0] = True
        path = [0]
        current = 0
        
        for _ in range(N - 1):
            neighbors = [v for v in adj[current] if not visited[v]]
            if not neighbors:
                break
            # Choose randomly, but prefer vertices with fewer unvisited neighbors (Warnsdorff-like)
            scored = []
            for v in neighbors:
                unvisited_count = sum(1 for w in adj[v] if not visited[w])
                scored.append((unvisited_count, v))
            scored.sort()
            # Pick from top few or random
            if rng.random() < 0.7:
                # Pick the one with fewest options (greedy)
                current = scored[0][1]
            else:
                current = rng.choice(neighbors)
            path.append(current)
            visited[current] = True
        
        if len(path) == N:
            # Check if we can close the cycle
            if 0 in adj[current]:
                best_cycle = path[:]
                success = True
                break
        # If we didn't complete, try again with different randomization
    
    return best_cycle

def two_opt_improve(cycle, K, max_iters=10000):
    """
    Try to improve a cycle so that all edges have index <= K using 2-opt moves.
    Returns improved cycle or None if stuck.
    """
    current = cycle[:]
    current_max = cycle_max_index(current)
    
    if current_max <= K:
        return current
    
    for _ in range(max_iters):
        improved = False
        for i in range(N - 1):
            for j in range(i + 2, N):
                if j == N - 1 and i == 0:
                    continue  # This would be the same edge
                # 2-opt: reverse the segment between i and j
                # Edges affected: (cycle[i-1], cycle[i]) and (cycle[j], cycle[(j+1)%N])
                # New edges: (cycle[i-1], cycle[j]) and (cycle[i], cycle[(j+1)%N])
                a = current[i - 1]
                b = current[i]
                c = current[j]
                d = current[(j + 1) % N]
                
                old_max1 = get_edge_pos(a, b)
                old_max2 = get_edge_pos(c, d)
                new_max1 = get_edge_pos(a, c)
                new_max2 = get_edge_pos(b, d)
                
                old_max = max(old_max1, old_max2)
                new_max = max(new_max1, new_max2)
                
                if new_max < old_max:
                    # Perform the move
                    current[i:j+1] = current[i:j+1][::-1]
                    improved = True
                    current_max = cycle_max_index(current)
                    if current_max <= K:
                        return current
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return current if current_max <= K else None

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s safety margin
    
    # Strategy: Binary search on K (the max edge index)
    # For each K, try to find a Hamilton cycle using only edges with index <= K
    
    # First, let's try a direct approach: start from a reasonable K and search
    
    # The minimum possible K is at least 24 (need 24 edges)
    # The maximum is 276
    
    # Let's try increasing K from a lower bound
    # First, find a lower bound: each vertex needs degree >= 2 in the prefix
    
    def min_degree_prefix(K):
        """Return the minimum degree in the prefix graph with edges up to K."""
        deg = [0] * N
        for (u, v), idx in EDGE_INDEX.items():
            if idx <= K:
                deg[u] += 1
                deg[v] += 1
        return min(deg)
    
    # Find the smallest K where min degree >= 2
    lo, hi = 24, TOTAL_EDGES
    while lo < hi:
        mid = (lo + hi) // 2
        if min_degree_prefix(mid) >= 2:
            hi = mid
        else:
            lo = mid + 1
    
    # Now try to find a Hamilton cycle for K = lo, lo+1, ...
    # Use randomized search for each K
    
    best_cycle = None
    best_K = None
    
    rng = random.Random(42)
    
    for K in range(lo, min(lo + 50, TOTAL_EDGES + 1)):
        if time.time() > deadline:
            break
        
        # Try randomized search for this K
        cycle = randomized_search(K, max_iters=20000, rng=rng)
        if cycle is not None:
            actual_max = cycle_max_index(cycle)
            if actual_max <= K:
                best_cycle = cycle
                best_K = actual_max
                break
        else:
            # Try a different approach: start from a full cycle and try to reduce max index
            pass
    
    if best_cycle is None:
        # Fallback: use a greedy construction
        # Build a cycle using the earliest available edges
        visited = [False] * N
        visited[0] = True
        cycle = [0]
        current = 0
        
        for _ in range(N - 1):
            # Find the neighbor with the smallest edge index
            best_neighbor = None
            best_idx = float('inf')
            for v in range(N):
                if not visited[v]:
                    idx = get_edge_pos(current, v)
                    if idx < best_idx:
                        best_idx = idx
                        best_neighbor = v
            
            if best_neighbor is None:
                break
            
            cycle.append(best_neighbor)
            visited[best_neighbor] = True
            current = best_neighbor
        
        if len(cycle) == N:
            best_cycle = cycle
            best_K = cycle_max_index(cycle)
    
    if best_cycle is None:
        # Last resort: return the identity permutation
        return list(range(N))
    
    # Try to improve with 2-opt
    if time.time() < deadline - 5:
        improved = two_opt_improve(best_cycle, best_K - 1, max_iters=5000)
        if improved is not None:
            best_cycle = improved
            best_K = cycle_max_index(best_cycle)
    
    return best_cycle
