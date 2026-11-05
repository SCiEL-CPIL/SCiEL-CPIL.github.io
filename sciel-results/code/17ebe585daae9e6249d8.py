import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']
TOTAL_EDGES = len(EDGE_ORDER)

# Precompute edge index for each pair
edge_idx = {}
for i, (u, v) in enumerate(EDGE_ORDER, 1):
    if u > v:
        u, v = v, u
    edge_idx[(u, v)] = i

def get_edge_index(u, v):
    if u > v:
        u, v = v, u
    return edge_idx.get((u, v), TOTAL_EDGES + 1)

def has_hamilton_cycle(K):
    """Check if graph with first K edges has a Hamilton cycle."""
    # Build adjacency as bitmasks
    adj = [0] * N
    for i in range(1, K + 1):
        u, v = EDGE_ORDER[i - 1]
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Necessary condition: all vertices have degree >= 2
    for v in range(N):
        if bin(adj[v]).count('1') < 2:
            return None
    
    # DFS with bitmask to find Hamilton cycle
    # Fix vertex 0 as start to break symmetry
    # We need to visit all vertices and return to 0
    full_mask = (1 << N) - 1
    
    # Use iterative DFS with memoization
    # State: (current_vertex, visited_mask)
    # We start at 0, need to visit all, and last vertex must connect back to 0
    
    from collections import deque
    
    # Memo: set of (vertex, mask) that have been explored
    memo = set()
    
    # Use a stack for DFS
    # Start: at vertex 0, visited = {0}
    start_mask = 1  # only vertex 0 visited
    
    # Try to find a path from 0 visiting all vertices, ending at a vertex adjacent to 0
    stack = [(0, start_mask, [0])]
    
    while stack:
        curr, mask, path = stack.pop()
        
        if len(path) == N:
            # Check if we can return to 0
            if adj[curr] & 1:  # edge from curr to 0
                return path[:]
            continue
        
        key = (curr, mask)
        if key in memo:
            continue
        memo.add(key)
        
        # Get unvisited neighbors
        unvisited = adj[curr] & ~mask
        # Try neighbors in order of fewest options first (heuristic)
        neighbors = []
        while unvisited:
            lsb = unvisited & (-unvisited)
            v = lsb.bit_length() - 1
            neighbors.append(v)
            unvisited ^= lsb
        
        # Sort by degree in remaining graph (fewer options first)
        remaining_mask = full_mask & ~mask
        neighbors.sort(key=lambda v: bin(adj[v] & remaining_mask).count('1'))
        
        for v in neighbors:
            new_mask = mask | (1 << v)
            # Pruning: check if any unvisited vertex (except possibly the last one) 
            # has no connection to unvisited vertices
            # Simple prune: if a vertex in remaining (not yet visited, not curr) has degree 0 in adj to (remaining | {curr})
            # This is expensive, so do a lighter check
            stack.append((v, new_mask, path + [v]))
    
    return None

def find_min_K():
    """Binary search for minimum K with Hamilton cycle."""
    # Lower bound: need at least N edges (a cycle has N edges)
    lo, hi = N, TOTAL_EDGES
    best_cycle = None
    best_K = TOTAL_EDGES + 1
    
    while lo <= hi:
        mid = (lo + hi) // 2
        cycle = has_hamilton_cycle(mid)
        if cycle is not None:
            best_cycle = cycle
            best_K = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    return best_K, best_cycle

def construct():
    deadline = time.time() + 80  # 80s deadline to be safe
    
    # First try binary search
    best_K, best_cycle = find_min_K()
    
    if best_cycle is not None:
        # Verify and return
        # The cycle is a list of vertices in order, with implicit closing edge
        # Return as permutation (the path without the last vertex, since closing edge is implicit)
        # Actually, the task says "a permutation of vertices 0..23, describing a Hamilton cycle with implicit closing edge"
        # So we return the full permutation of 24 vertices
        return best_cycle
    
    # Fallback: return identity
    return list(range(N))

# EVOLVE-BLOCK-END
