import time
import sys
from collections import defaultdict

N = 24
EDGE_ORDER = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

# Precompute edge positions
edge_pos = {}
for i, (u, v) in enumerate(EDGE_ORDER):
    a, b = min(u, v), max(u, v)
    edge_pos[(a, b)] = i + 1

def find_hamilton_cycle(adj_mask, n, deadline):
    """
    Finds a Hamilton cycle using bitmask DP with backtracking and constraint propagation.
    adj_mask[v] is a bitmask of neighbors of v.
    Returns the list of vertices in the cycle or None.
    """
    # Quick checks
    for v in range(n):
        if adj_mask[v].bit_count() < 2:
            return None
    
    # Fix vertex 0 as start.
    start = 0
    visited_mask = 1 << start
    path = [start]
    adj_to_start = adj_mask[start]
    
    def backtrack(current, mask, depth):
        if time.time() > deadline:
            return None
        
        if depth == n:
            if (1 << current) & adj_to_start:
                return path[:]
            return None
        
        # Get unvisited neighbors of current
        unvisited = ~mask & ((1 << n) - 1)
        candidates = adj_mask[current] & unvisited
        
        if candidates == 0:
            return None
        
        # Constraint Propagation: Connectivity Check
        # The set of nodes S = {current} U unvisited must be connected in the graph.
        # If not, we cannot visit all unvisited nodes in a single path starting from current.
        if n - depth > 1:
            # BFS from current within S
            target_mask = unvisited | (1 << current)
            visited_bfs = 1 << current
            stack = [current]
            while stack:
                u = stack.pop()
                neighbors = adj_mask[u] & target_mask & ~visited_bfs
                while neighbors:
                    low = neighbors & (-neighbors)
                    v = low.bit_length() - 1
                    neighbors ^= low
                    visited_bfs |= low
                    stack.append(v)
            
            if visited_bfs != target_mask:
                return None

        # Heuristic: Warnsdorff
        cands = []
        c = candidates
        while c:
            low = c & (-c)
            v = low.bit_length() - 1
            c ^= low
            unvisited_without_v = unvisited & ~(1 << v)
            deg = (adj_mask[v] & unvisited_without_v).bit_count()
            cands.append((deg, v))
        
        cands.sort(key=lambda x: x[0])
        
        for deg, nxt in cands:
            path.append(nxt)
            result = backtrack(nxt, mask | (1 << nxt), depth + 1)
            if result:
                return result
            path.pop()
        
        return None
    
    return backtrack(start, visited_mask, 1)


def check_hamilton_cycle_exists(adj_mask, n, deadline):
    """Returns True if a Hamilton cycle exists, False otherwise (or timeout)."""
    # Quick degree check
    for v in range(n):
        if adj_mask[v].bit_count() < 2:
            return False
    
    # Connectivity check using BFS
    visited = 1  # start at 0
    stack = [0]
    while stack:
        u = stack.pop()
        neighbors = adj_mask[u] & ~visited
        while neighbors:
            low = neighbors & (-neighbors)
            v = low.bit_length() - 1
            neighbors ^= low
            visited |= low
            stack.append(v)
    if visited != (1 << n) - 1:
        return False
    
    # Try to find a cycle
    cycle = find_hamilton_cycle(adj_mask, n, deadline)
    return cycle is not None


def construct():
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s margin
    
    n = N
    total_edges = n * (n - 1) // 2
    
    # Build adjacency masks for a given threshold
    def build_adj_mask(threshold):
        adj = [0] * n
        for i, (u, v) in enumerate(EDGE_ORDER):
            if i + 1 <= threshold:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj
    
    # Binary search for minimum threshold
    lo, hi = 24, total_edges
    best_cycle = None
    best_threshold = total_edges
    
    # First, find a valid cycle at a high threshold to have a fallback
    # Start from the top and work down
    # Actually, let's do binary search properly
    
    # But first, let's establish an upper bound with a quick check
    # Try progressively lower thresholds
    
    # Strategy: 
    # 1. Find any valid cycle (upper bound)
    # 2. Binary search for the minimum
    
    # For step 1, try threshold = total_edges (all edges, guaranteed cycle)
    # But we want a good cycle, so let's be smarter.
    
    # Actually, for the complete graph, any permutation is a valid cycle.
    # The score is determined by the max edge position.
    # So we just need to find the minimum T such that a Hamilton cycle exists
    # using only edges with position <= T.
    
    # Let's do binary search
    # But we need to be careful: the backtracking might be slow for some thresholds.
    
    # Let's first find an upper bound by trying some thresholds
    # Start with a reasonable guess and adjust
    
    # Try thresholds from low to high until we find one that works
    # But this could be slow. Let's do binary search.
    
    # First, verify that total_edges works (it should, since it's complete graph)
    adj_full = build_adj_mask(total_edges)
    cycle = find_hamilton_cycle(adj_full, n, deadline)
    if cycle:
        best_cycle = cycle
        best_threshold = total_edges
    else:
        # Shouldn't happen, but fallback
        best_cycle = list(range(n))
    
    # Now binary search for lower threshold
    lo, hi = 24, best_threshold - 1
    
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        adj = build_adj_mask(mid)
        
        cycle = find_hamilton_cycle(adj, n, deadline)
        
        if cycle:
            best_cycle = cycle
            best_threshold = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    return best_cycle


# Test locally
if __name__ == "__main__":
    c = construct()
    max_pos = 0
    for i in range(N):
        u = c[i]
        v = c[(i+1)%N]
        a, b = min(u, v), max(u, v)
        p = edge_pos[(a, b)]
        if p > max_pos:
            max_pos = p
    print(f"Cycle: {c}")
    print(f"Score: {max_pos}")
