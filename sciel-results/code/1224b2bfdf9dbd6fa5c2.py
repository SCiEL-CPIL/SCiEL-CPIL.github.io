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

def get_pos(u, v):
    a, b = min(u, v), max(u, v)
    return edge_pos[(a, b)]

def find_hamilton_cycle_backtrack(adj, n, deadline, start_time):
    """
    Finds a Hamilton cycle using backtracking with pruning.
    adj is a list of sets for fast membership.
    Returns the list of vertices in the cycle or None.
    """
    # Quick degree check
    for v in range(n):
        if len(adj[v]) < 2:
            return None
    
    # Connectivity check using BFS
    visited = [False] * n
    stack = [0]
    visited[0] = True
    count = 0
    while stack:
        u = stack.pop()
        count += 1
        for w in adj[u]:
            if not visited[w]:
                visited[w] = True
                stack.append(w)
    if count != n:
        return None
    
    # Precompute neighbor bitmasks and sorted neighbors
    neighbor_mask = [0] * n
    for v in range(n):
        for w in adj[v]:
            neighbor_mask[v] |= (1 << w)
    
    # Sort neighbors by degree for better pruning (Warnsdorff's rule)
    sorted_neighbors = []
    for v in range(n):
        neighbors = sorted(adj[v], key=lambda x: len(adj[x]))
        sorted_neighbors.append(neighbors)
    
    def backtrack2(mask, path):
        if time.time() > deadline:
            return None
        
        if len(path) == n:
            last = path[-1]
            if 0 in adj[last]:
                return path[:]
            return None
        
        last = path[-1]
        
        # If only 1 vertex left, check directly
        if len(path) == n - 1:
            for v in range(n):
                if not (mask >> v) & 1:
                    if v in adj[last] and 0 in adj[v]:
                        return path + [v]
            return None
        
        # Try neighbors in order of increasing degree
        for nxt in sorted_neighbors[last]:
            if (mask >> nxt) & 1:
                continue
            
            new_mask = mask | (1 << nxt)
            
            # Pruning: check for isolated unvisited vertices
            is_dead_end = False
            unvisited_count = 0
            for w in range(n):
                if (new_mask >> w) & 1:
                    continue
                unvisited_count += 1
                has_unvisited = False
                for nb in adj[w]:
                    if not (new_mask >> nb) & 1:
                        has_unvisited = True
                        break
                if not has_unvisited:
                    if unvisited_count > 1:
                        is_dead_end = True
                        break
                    else:
                        # Only 1 unvisited, must connect to last and 0
                        if not (w in adj[last] and 0 in adj[w]):
                            is_dead_end = True
                        break
            
            if is_dead_end:
                continue
            
            result = backtrack2(new_mask, path + [nxt])
            if result is not None:
                return result
        
        return None
    
    return backtrack2(1, [0])

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s margin
    
    n = N
    total_edges = n * (n - 1) // 2
    
    # Binary search for the minimum threshold T
    lo, hi = 24, total_edges
    best_cycle = None
    
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        
        # Build subgraph with edges at position <= mid
        adj = [set() for _ in range(n)]
        for (u, v), pos in edge_pos.items():
            if pos <= mid:
                adj[u].add(v)
                adj[v].add(u)
        
        # Check for Hamilton cycle
        cycle = find_hamilton_cycle_backtrack(adj, n, deadline, start_time)
        
        if cycle is not None:
            best_cycle = cycle
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_cycle is not None:
        return best_cycle
    
    # Fallback: use all edges
    if time.time() < deadline:
        adj = [set() for _ in range(n)]
        for (u, v), pos in edge_pos.items():
            adj[u].add(v)
            adj[v].add(u)
        
        cycle = find_hamilton_cycle_backtrack(adj, n, start_time + 5, start_time)
        if cycle:
            return cycle
    
    return list(range(n))

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
