"""Find Hamilton cycle with minimal max edge arrival index."""
# EVOLVE-BLOCK-START
import time
import sys

N = 24
TIME_LIMIT = 85  # seconds, leaving margin for evaluator overhead
EDGE_ORDER = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

# Precompute edge positions
edge_pos = {}
for i, (u, v) in enumerate(EDGE_ORDER, 1):
    edge_pos[(u, v)] = i
    edge_pos[(v, u)] = i

def popcount(x):
    """Count bits in integer."""
    return bin(x).count('1')

def iter_bits(mask):
    """Iterate over set bits in mask, yielding the bit index."""
    while mask:
        lb = mask & (-mask)
        yield lb.bit_length() - 1
        mask ^= lb

def build_adj_mask(K):
    """Return list of bitmasks for each vertex, using edges with position <= K."""
    masks = [0] * N
    for i in range(1, K + 1):
        u, v = EDGE_ORDER[i-1]
        masks[u] |= (1 << v)
        masks[v] |= (1 << u)
    return masks

def is_connected(masks):
    """Check if graph with adjacency masks is connected."""
    visited = 0
    start = 0
    # Find first vertex with degree > 0
    while start < N and masks[start] == 0:
        start += 1
    if start == N:
        return N == 1 # Empty graph edge case, not applicable here
    
    stack = [start]
    visited |= (1 << start)
    
    while stack:
        v = stack.pop()
        neighbors = masks[v] & ~visited
        while neighbors:
            lb = neighbors & (-neighbors)
            u = lb.bit_length() - 1
            visited |= lb
            stack.append(u)
            neighbors ^= lb
            
    return visited == (1 << N) - 1

def has_bridges(masks):
    """Check if graph has any bridges. If yes, no Hamilton cycle exists."""
    # Tarjan's algorithm for bridges
    # Since N is small (24), we can use a simple DFS
    disc = [-1] * N
    low = [-1] * N
    parent = [-1] * N
    time_counter = [0]
    has_bridge = [False]
    
    def dfs(u):
        if has_bridge[0]:
            return
        disc[u] = low[u] = time_counter[0]
        time_counter[0] += 1
        children = 0
        neighbors = masks[u]
        while neighbors:
            lb = neighbors & (-neighbors)
            v = lb.bit_length() - 1
            neighbors ^= lb
            
            if disc[v] == -1:
                parent[v] = u
                children += 1
                dfs(v)
                if has_bridge[0]:
                    return
                low[u] = min(low[u], low[v])
                
                # If low[v] > disc[u], then (u, v) is a bridge
                if low[v] > disc[u]:
                    has_bridge[0] = True
                    return
            elif v != parent[u]:
                low[u] = min(low[u], disc[v])
                
    for i in range(N):
        if disc[i] == -1:
            if i > 0: 
                return True # Disconnected, handled by is_connected, but just in case
            dfs(i)
            if has_bridge[0]:
                return True
    return False

def has_hamilton_cycle(K, start=0, deadline=None):
    """Check if there's a Hamilton cycle using only edges with position <= K.
    Returns the cycle as a list of vertices, or None."""
    
    # 1. Build masks
    masks = build_adj_mask(K)
    
    # 2. Quick checks
    # Every vertex must have degree >= 2
    for v in range(N):
        if popcount(masks[v]) < 2:
            return None
            
    # Graph must be connected
    if not is_connected(masks):
        return None
        
    # Graph must be 2-connected (no bridges)
    if has_bridges(masks):
        return None
        
    # 3. DFS for Hamiltonian Cycle
    full_mask = (1 << N) - 1
    start_bit = 1 << start
    
    sys.setrecursionlimit(10000)
    
    # Memo: (vertex, visited) -> False
    failed = set()
    
    def dfs(v, visited, path):
        if deadline is not None and time.time() > deadline:
            return None
            
        if len(path) == N:
            # Check if v connects back to start
            if masks[v] & start_bit:
                return path[:]
            return None
        
        # Pruning: Check if any unvisited vertex is isolated from the rest
        # (excluding current v which is the only connection point for the path end)
        # Actually, a stronger check:
        # Any unvisited vertex u must have at least 2 neighbors in (unvisited U {v}) 
        # OR if u is the last one, 1 neighbor in unvisited and 1 to start?
        # Simple check: u must have >= 1 neighbor in (unvisited U {v})
        # And if u is the only unvisited, it must connect to start.
        
        unvisited = full_mask & ~visited
        # Optimization: Only check if path length is small or medium?
        # For N=24, this loop is 24 iterations. Acceptable.
        for u in iter_bits(unvisited):
            # Neighbors of u in (unvisited U {v})
            avail_u = masks[u] & (unvisited | (1 << v))
            # If u is the last vertex to be visited (len(path) == N-1), 
            # it needs to connect to start.
            # If not last, it needs to connect to at least one other unvisited vertex 
            # or v (if v is the only connection back).
            # Actually, if avail_u is 0, it's a dead end.
            if avail_u == 0:
                return None
        
        available = masks[v] & ~visited
        if available == 0:
            return None
            
        # Warnsdorff's heuristic: sort neighbors by number of onward connections
        neighbors = list(iter_bits(available))
        
        # Special case: next-to-last vertex
        if len(path) == N - 1:
            # The last vertex must be able to connect back to start
            filtered = [nb for nb in neighbors if masks[nb] & start_bit]
            if not filtered:
                return None
            neighbors = filtered
            
        # Sort by onward degree (fewest first)
        # onward degree = popcount(masks[nb] & ~(visited | (1 << nb)))
        neighbors.sort(key=lambda nb: popcount(masks[nb] & ~(visited | (1 << nb))))
        
        for nb in neighbors:
            new_visited = visited | (1 << nb)
            key = (nb, new_visited)
            if key in failed:
                continue
            result = dfs(nb, new_visited, path + [nb])
            if result is not None:
                return result
            failed.add(key)
            
        return None
    
    return dfs(start, start_bit, [start])

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - 5  # safety margin
    
    # Binary search for minimum K
    lo, hi = 24, len(EDGE_ORDER)
    best_cycle = None
    
    while lo <= hi:
        if time.time() > deadline - 3:
            break
        mid = (lo + hi) // 2
        cycle = has_hamilton_cycle(mid, start=0, deadline=deadline)
        if cycle is not None:
            best_cycle = cycle
            hi = mid - 1
        else:
            lo = mid + 1
            
    if best_cycle is not None:
        return best_cycle
        
    # Fallback: try larger K values directly if binary search failed or timed out
    for K in range(50, len(EDGE_ORDER) + 1, 5):
        if time.time() > deadline - 2:
            break
        cycle = has_hamilton_cycle(K, start=0, deadline=deadline)
        if cycle is not None:
            best_cycle = cycle
            break
            
    if best_cycle is not None:
        return best_cycle
        
    # Last resort: identity permutation
    return list(range(N))
# EVOLVE-BLOCK-END
