"""Find Hamilton cycle minimizing max edge arrival index in K_24."""
import time
import random
import sys

# Edge order from the task
edge_order = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

N = 24
TOTAL_EDGES = len(edge_order)

# Build edge index map: edge (u,v) -> 1-based arrival index
edge_index = {}
for i, (u, v) in enumerate(edge_order, 1):
    key = (min(u, v), max(u, v))
    edge_index[key] = i

def get_edge_idx(u, v):
    key = (min(u, v), max(u, v))
    return edge_index[key]

def build_adj(K):
    """Build adjacency list for edges with index <= K."""
    adj = [set() for _ in range(N)]
    for i in range(1, K + 1):
        u, v = edge_order[i - 1]
        adj[u].add(v)
        adj[v].add(u)
    return adj

def has_hamilton_cycle(adj, deadline):
    """Check if graph with adjacency list 'adj' has a Hamilton cycle.
    Returns the cycle as a list of vertices, or None."""
    # Fix vertex 0 as start
    start = 0
    n = N
    
    # Quick check: all vertices must have degree >= 2
    for v in range(n):
        if len(adj[v]) < 2:
            return None
    
    # Backtracking DFS
    visited = [False] * n
    path = [start]
    visited[start] = True
    remaining = set(range(n))
    remaining.discard(start)
    
    sys.setrecursionlimit(10000)
    
    def dfs(current, depth):
        if time.time() > deadline:
            return None
        
        if depth == n - 1:
            # Check if current can connect back to start
            if start in adj[current]:
                return list(path)
            return None
        
        # Pruning: check if any remaining vertex is isolated
        # (no edge to current and no edges among remaining that would allow completion)
        # Simpler: check if current has any unvisited neighbor
        neighbors = [v for v in adj[current] if not visited[v]]
        if not neighbors:
            return None
        
        # Degree pruning: for each remaining vertex, check if it still has at least 2 connections
        # in the subgraph induced by {current} + remaining (considering path)
        # This is expensive, so do a lighter check:
        # If a remaining vertex has 0 unvisited neighbors and isn't adjacent to current, prune
        for v in remaining:
            if v == current:
                continue
            # Count unvisited neighbors of v (excluding current since current is visited)
            unvisited_neighbors = 0
            for w in adj[v]:
                if not visited[w] and w != current:
                    unvisited_neighbors += 1
            # v needs at least 2 connections in the final cycle
            # One could be to current (if adjacent), one to another unvisited
            # This is a weak check; skip for speed
            pass
        
        # Order neighbors by degree (fewest first) for better pruning
        neighbors.sort(key=lambda v: len([w for w in adj[v] if not visited[w]]))
        
        for nxt in neighbors:
            visited[nxt] = True
            path.append(nxt)
            remaining.discard(nxt)
            
            result = dfs(nxt, depth + 1)
            if result is not None:
                return result
            
            visited[nxt] = False
            path.pop()
            remaining.add(nxt)
        
        return None
    
    return dfs(start, 0)

def binary_search_cycle():
    """Binary search for minimum K with Hamilton cycle."""
    lo = N  # minimum edges needed for a cycle is N
    hi = TOTAL_EDGES
    
    best_cycle = None
    best_K = hi + 1
    
    while lo <= hi:
        mid = (lo + hi) // 2
        adj = build_adj(mid)
        cycle = has_hamilton_cycle(adj, deadline)
        if cycle is not None:
            best_cycle = cycle
            best_K = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    return best_cycle, best_K

def construct():
    global deadline
    start_time = time.time()
    deadline = start_time + 75  # 75s deadline, 15s safety margin
    
    # First try binary search
    try:
        cycle, K = binary_search_cycle()
        if cycle is not None:
            return cycle
    except Exception:
        pass
    
    # Fallback: try a greedy/heuristic approach
    # Build a cycle by always picking the earliest available edge
    # Start with vertex 0, greedily add vertices
    
    # Alternative fallback: use the identity permutation and hope
    # But let's try a simple 2-opt improvement
    cycle = list(range(N))
    
    # Compute current score
    def score(cyc):
        max_idx = 0
        for i in range(len(cyc)):
            u = cyc[i]
            v = cyc[(i + 1) % len(cyc)]
            idx = get_edge_idx(u, v)
            if idx > max_idx:
                max_idx = idx
        return max_idx
    
    current_score = score(cycle)
    
    # 2-opt local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N - 1):
            for j in range(i + 2, N):
                if i == 0 and j == N - 1:
                    continue
                # Try reversing segment [i+1..j]
                new_cycle = cycle[:i+1] + cycle[i+1:j+1][::-1] + cycle[j+1:]
                ns = score(new_cycle)
                if ns < current_score:
                    cycle = new_cycle
                    current_score = ns
                    improved = True
    
    return cycle

# Pre-compute for speed
deadline = time.time() + 75
