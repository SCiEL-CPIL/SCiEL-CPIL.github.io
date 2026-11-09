import time
import sys
from itertools import combinations

# Precompute edge positions
N = 24
EDGE_ORDER = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

# Build edge position map: (min(u,v), max(u,v)) -> 1-based position
edge_pos = {}
for i, (u, v) in enumerate(EDGE_ORDER):
    a, b = min(u, v), max(u, v)
    edge_pos[(a, b)] = i + 1

def get_pos(u, v):
    a, b = min(u, v), max(u, v)
    return edge_pos[(a, b)]

def has_hamilton_cycle(adj, n):
    """Check if graph with adjacency list has a Hamilton cycle using backtracking."""
    # Quick checks
    for v in range(n):
        if len(adj[v]) < 2:
            return False
    
    # Check connectivity
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
        return False
    
    # Backtracking Hamilton cycle search
    # Fix vertex 0 as start to reduce symmetry
    # Path: 0 -> ... -> 0
    path = [0]
    used = [False] * n
    used[0] = True
    
    def backtrack():
        if len(path) == n:
            # Check if last vertex connects back to 0
            last = path[-1]
            if 0 in adj[last]:
                return True
            return False
        
        last = path[-1]
        # Try neighbors of last, ordered by degree for better pruning
        neighbors = sorted(adj[last], key=lambda x: len(adj[x]))
        for nxt in neighbors:
            if not used[nxt]:
                # Pruning: if nxt is not 0 and not the last step, 
                # check that 0 still has at least 2 available neighbors (or will)
                # Simple pruning: remaining unvisited vertices must be connected
                # to each other and to last
                used[nxt] = True
                path.append(nxt)
                
                # Check: for the closing edge, 0 must be reachable from final vertex
                # Prune: if we're at step n-1 (one more to go), next must connect to 0
                if len(path) == n - 1:
                    # The last unvisited vertex must connect to both nxt and 0
                    remaining = [v for v in range(n) if not used[v]]
                    if len(remaining) == 1:
                        r = remaining[0]
                        if r not in adj[nxt] or 0 not in adj[r]:
                            path.pop()
                            used[nxt] = False
                            continue
                
                if backtrack():
                    return True
                
                path.pop()
                used[nxt] = False
        
        return False
    
    return backtrack()

def find_cycle_in_subgraph(adj, n):
    """Find a Hamilton cycle in the given adjacency list, return list of vertices or None."""
    for v in range(n):
        if len(adj[v]) < 2:
            return None
    
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
    
    path = [0]
    used = [False] * n
    used[0] = True
    
    def backtrack():
        if len(path) == n:
            last = path[-1]
            if 0 in adj[last]:
                return path[:]
            return None
        
        last = path[-1]
        neighbors = sorted(adj[last], key=lambda x: len(adj[x]))
        for nxt in neighbors:
            if not used[nxt]:
                used[nxt] = True
                path.append(nxt)
                
                if len(path) == n - 1:
                    remaining = [v for v in range(n) if not used[v]]
                    if len(remaining) == 1:
                        r = remaining[0]
                        if r not in adj[nxt] or 0 not in adj[r]:
                            path.pop()
                            used[nxt] = False
                            continue
                
                result = backtrack()
                if result is not None:
                    return result
                
                path.pop()
                used[nxt] = False
        
        return None
    
    return backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s margin
    
    n = N
    total_edges = n * (n - 1) // 2  # 276
    
    # Binary search for minimum threshold T
    # Lower bound: need at least n edges in any Hamilton cycle, but they must form a cycle
    # The minimum possible is n (24), but practically higher
    # Upper bound: total_edges
    
    best_cycle = None
    best_score = total_edges + 1
    
    # First, try to find a good cycle with greedy approach
    # Sort all edges by position
    sorted_edges = sorted(EDGE_ORDER, key=lambda e: edge_pos[(min(e[0],e[1]), max(e[0],e[1]))])
    
    # Try incremental threshold search
    # Start from a reasonable lower bound
    # A Hamilton cycle needs 24 edges. The 24th edge in the list is at position 24.
    # But we need them to form a cycle, so likely need more.
    
    lo, hi = 24, total_edges
    
    # Binary search
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        
        # Build subgraph with edges at position <= mid
        adj = [[] for _ in range(n)]
        for (u, v), pos in edge_pos.items():
            if pos <= mid:
                adj[u].append(v)
                adj[v].append(u)
        
        # Check for Hamilton cycle
        cycle = find_cycle_in_subgraph(adj, n)
        
        if cycle is not None:
            best_cycle = cycle
            best_score = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    # If we found a cycle, return it
    if best_cycle is not None:
        return best_cycle
    
    # Fallback: try greedy construction
    # Build cycle greedily using earliest edges
    if time.time() < deadline:
        adj_full = [[] for _ in range(n)]
        for (u, v) in EDGE_ORDER:
            adj_full[u].append(v)
            adj_full[v].append(u)
        
        # Sort neighbors by edge position
        for v in range(n):
            adj_full[v].sort(key=lambda x: get_pos(v, x))
        
        # Backtracking with full graph
        path = [0]
        used = [False] * n
        used[0] = True
        
        def backtrack_greedy():
            if time.time() > deadline:
                return None
            if len(path) == n:
                last = path[-1]
                if 0 in adj_full[last]:
                    return path[:]
                return None
            
            last = path[-1]
            for nxt in adj_full[last]:
                if not used[nxt]:
                    used[nxt] = True
                    path.append(nxt)
                    
                    if len(path) == n - 1:
                        remaining = [v for v in range(n) if not used[v]]
                        if len(remaining) == 1:
                            r = remaining[0]
                            if r not in adj_full[nxt] or 0 not in adj_full[r]:
                                path.pop()
                                used[nxt] = False
                                continue
                    
                    result = backtrack_greedy()
                    if result is not None:
                        return result
                    
                    path.pop()
                    used[nxt] = False
            
            return None
        
        cycle = backtrack_greedy()
        if cycle is not None:
            return cycle
    
    # Final fallback: identity permutation
    return list(range(n))
