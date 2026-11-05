import sys
import time
import random

sys.setrecursionlimit(10000)

# Edge order from the task
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = 24
EDGE_ORDER = P['edge_order']
NUM_EDGES = len(EDGE_ORDER)

# Precompute edge index
edge_idx = {}
for i, (u, v) in enumerate(EDGE_ORDER):
    edge_idx[(u, v)] = i + 1
    edge_idx[(v, u)] = i + 1


def find_hamilton_cycle_fast(k, deadline):
    """
    Try to find a Hamilton cycle in the graph formed by the first k edges.
    Returns the cycle as a list of vertices, or None if not found or timeout.
    Uses aggressive backtracking with Warnsdorff's heuristic.
    """
    # Build adjacency as lists
    adj = [[] for _ in range(N)]
    for i in range(k):
        u, v = EDGE_ORDER[i]
        adj[u].append(v)
        adj[v].append(u)
    
    # Quick degree check
    for i in range(N):
        if len(adj[i]) < 2:
            return None
    
    # Connectivity check using BFS
    visited = [False] * N
    stack = [0]
    visited[0] = True
    count = 0
    while stack:
        u = stack.pop()
        count += 1
        if time.time() > deadline:
            return None
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                stack.append(v)
    if count != N:
        return None
    
    # Sort adjacency by degree (ascending) for Warnsdorff
    adj_sorted = []
    for i in range(N):
        adj_sorted.append(sorted(adj[i], key=lambda x: len(adj[x])))
    
    # Backtracking with Warnsdorff's heuristic
    path = [0]
    visited = [False] * N
    visited[0] = True
    
    def dfs(u, depth):
        if time.time() > deadline:
            return False
        
        if depth == N:
            # Check if we can close the cycle
            if 0 in adj[u]:
                return True
            return False
        
        # Get unvisited neighbors sorted by their degree (Warnsdorff)
        candidates = []
        for v in adj_sorted[u]:
            if not visited[v]:
                # Count unvisited neighbors of v (for Warnsdorff)
                unvisited_count = 0
                for w in adj[v]:
                    if not visited[w]:
                        unvisited_count += 1
                candidates.append((unvisited_count, v))
        
        # Sort by unvisited count (Warnsdorff: go to vertex with fewest options)
        candidates.sort()
        
        for _, v in candidates:
            if time.time() > deadline:
                return False
            
            # Pruning: if v is not the last vertex, it must have at least 
            # one unvisited neighbor (or be able to connect to 0 if last)
            if depth < N - 1:
                unvisited_after = 0
                for w in adj[v]:
                    if not visited[w] and w != u:
                        unvisited_after += 1
                if unvisited_after == 0:
                    continue
            
            path.append(v)
            visited[v] = True
            
            if dfs(v, depth + 1):
                return True
            
            path.pop()
            visited[v] = False
        
        return False
    
    if dfs(0, 1):
        return path
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin for the 90s limit
    
    n = N
    min_k = n  # Need at least n edges for a cycle
    max_k = NUM_EDGES
    
    best_cycle = None
    best_k = max_k
    
    # Try to find the minimum K with a Hamilton cycle
    # Use a smarter search: try values that are likely to work
    # Start from a reasonable estimate and expand
    
    # First, let's try a range of K values
    # The minimum possible is N (24 edges for a cycle)
    # Let's try from low to high, but skip some values for speed
    
    # Try binary search first to get a rough bound
    low = min_k
    high = max_k
    
    # First pass: binary search to find approximate minimum
    while low <= high and time.time() < deadline:
        mid = (low + high) // 2
        cycle = find_hamilton_cycle_fast(mid, deadline)
        if cycle is not None:
            best_cycle = cycle
            best_k = mid
            high = mid - 1
        else:
            low = mid + 1
    
    # Now refine: try values near best_k to see if we can do better
    if best_cycle is not None and time.time() < deadline:
        # Try a few values below best_k
        for k in range(max(min_k, best_k - 5), best_k):
            if time.time() > deadline:
                break
            cycle = find_hamilton_cycle_fast(k, deadline)
            if cycle is not None:
                best_cycle = cycle
                best_k = k
                break
    
    if best_cycle is None:
        # Fallback: return identity cycle
        return list(range(n))
    
    return best_cycle
