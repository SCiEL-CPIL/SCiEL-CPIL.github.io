import time
import sys
from collections import deque

# EVOLVE-BLOCK-START

n = 24
edge_order = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

# Precompute arrival index (0-based) for each edge
arrival = {}
for idx, (u, v) in enumerate(edge_order):
    key = (min(u, v), max(u, v))
    arrival[key] = idx

def get_arrival(u, v):
    key = (min(u, v), max(u, v))
    return arrival[key]

def compute_score(cycle):
    """Compute the score (max 1-based arrival index) of a Hamilton cycle."""
    max_idx = 0
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        idx = get_arrival(u, v) + 1  # 1-based
        if idx > max_idx:
            max_idx = idx
    return max_idx

def find_hamiltonian_cycle(k, time_limit):
    """
    Find a Hamiltonian cycle in the subgraph formed by the first k edges.
    Returns the cycle as a list of vertices if found, None otherwise.
    """
    # Build adjacency list for G_k
    adj = [[] for _ in range(n)]
    for i in range(k):
        u, v = edge_order[i]
        adj[u].append(v)
        adj[v].append(u)
    
    # Connectivity check
    visited_conn = [False] * n
    stack = [0]
    visited_conn[0] = True
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if not visited_conn[v]:
                visited_conn[v] = True
                stack.append(v)
    if not all(visited_conn):
        return None
        
    # Check degree constraints
    for i in range(n):
        if len(adj[i]) < 2:
            return None
            
    deadline = time.time() + time_limit
    visited = [False] * n
    path = [0]
    visited[0] = True
    
    def is_feasible():
        # Check unvisited vertices
        for u in range(n):
            if not visited[u]:
                if len(adj[u]) < 2:
                    return False
        # Check if the current path is a dead end
        if len(path) < n:
            last = path[-1]
            has_unvisited_neighbor = False
            for nb in adj[last]:
                if not visited[nb]:
                    has_unvisited_neighbor = True
                    break
            if not has_unvisited_neighbor:
                return False
        return True

    def backtrack():
        if time.time() > deadline:
            return None
        
        if len(path) == n:
            last = path[-1]
            if 0 in adj[last]:
                return path[:]
            return None
        
        if not is_feasible():
            return None
            
        current = path[-1]
        
        # Heuristic: Warnsdorff's rule
        # Choose the neighbor with the fewest unvisited neighbors
        candidates = []
        for nb in adj[current]:
            if not visited[nb]:
                onward = 0
                for nbn in adj[nb]:
                    if not visited[nbn] and nbn != current:
                        onward += 1
                candidates.append((onward, nb))
                
        candidates.sort()
        
        for _, nb in candidates:
            visited[nb] = True
            path.append(nb)
            
            result = backtrack()
            if result is not None:
                return result
            
            path.pop()
            visited[nb] = False
            
        return None
        
    return backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    # Default cycle
    default_cycle = list(range(n))
    best_score = compute_score(default_cycle)
    best_cycle = default_cycle
    
    # Binary search for the minimum k
    # Lower bound: 23 edges is not enough for a cycle on 24 vertices? 
    # A cycle needs n edges. So k must be at least 24.
    # Upper bound: 276 (all edges)
    lo = 24
    hi = 276
    
    # We know best_score is the 1-based index of the max edge in default cycle.
    # We want to find the smallest k such that a cycle exists.
    # The score will be k (since the cycle uses edges from the first k, the max index is k-1, 1-based is k).
    # Wait, if we find a cycle in G_k, its score is <= k.
    # We want to minimize the score.
    
    # First, check if the default cycle is already good.
    # Then binary search for smaller k.
    
    # Actually, just linear scan from low to high might be faster if the answer is low.
    # But binary search is safer for worst case.
    # Let's try binary search.
    
    # Optimization: We can just check k from 24 upwards.
    # The number of edges in a random graph G(n, p) needed for Hamiltonicity is around n/2 * log n.
    # For n=24, n log n ~ 24 * 3.2 = 77.
    # So the answer is likely around 80-100.
    # Linear scan from 24 to 100 is only 76 steps.
    # Each step is a DFS. DFS is fast if it fails early or succeeds early.
    
    # Let's do a linear scan, but with a time limit per check.
    # If a check takes too long, we skip to a higher k? No, that breaks binary search logic.
    # But for linear scan, if k is too low, DFS fails fast (degree < 2).
    # So linear scan is good.
    
    lo = 24
    hi = 276
    best_cycle = None
    best_k = 276

    while lo <= hi:
        if time.time() > deadline - 10:
            break
            
        mid = (lo + hi) // 2
        
        # Use a generous time limit for binary search steps
        t_limit = 5.0
        
        cycle = find_hamiltonian_cycle(mid, t_limit)
        if cycle is not None:
            best_cycle = cycle
            best_k = mid
            hi = mid - 1
        else:
            lo = mid + 1
            
    if best_cycle is not None:
        return best_cycle
    else:
        # Fallback: return the default cycle or try a high k
        # We should have found one by k=276
        return default_cycle

# EVOLVE-BLOCK-END
