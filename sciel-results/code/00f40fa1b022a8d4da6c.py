"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

_N = 24

# Precompute the edge order from the given data
_EDGE_DATA = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

# Build arrival index and edge list
_EDGE_INDEX = {}
for i, (u, v) in enumerate(_EDGE_DATA):
    if u > v:
        u, v = v, u
    _EDGE_INDEX[(u, v)] = i + 1

def _get_edge_index(u, v):
    if u > v:
        u, v = v, u
    return _EDGE_INDEX.get((u, v), float('inf'))

def _find_hamiltonian_cycle(K, deadline):
    """
    Find a Hamiltonian cycle using only edges with index <= K.
    Returns the cycle (list of vertices) or None if not found or timeout.
    """
    # Build adjacency list for allowed edges
    adj = {i: set() for i in range(_N)}
    for i, (u, v) in enumerate(_EDGE_DATA):
        if i + 1 > K:
            break
        adj[u].add(v)
        adj[v].add(u)
    
    # Check basic connectivity/degree
    for i in range(_N):
        if len(adj[i]) < 2:
            return None

    start = 0
    path = [start]
    visited = {start}
    
    # Heuristic: Order neighbors by degree (MRV)
    # We will sort dynamically in the recursion step
    
    def dfs(current, path, visited):
        if time.time() > deadline:
            return None
        
        if len(path) == _N:
            # Check if we can close the cycle
            if start in adj[current]:
                return list(path)
            return None
        
        current_idx = len(path)
        # Get unvisited neighbors
        candidates = []
        for neighbor in adj[current]:
            if neighbor not in visited:
                candidates.append(neighbor)
        
        if not candidates:
            return None
            
        # Sort candidates by degree in the induced subgraph of unvisited + current?
        # Simpler: Sort by global degree in allowed graph (lower degree first)
        # This is a static heuristic, but effective for pruning.
        candidates.sort(key=lambda x: len(adj[x]))
        
        for next_v in candidates:
            # Forward checking:
            # If we visit next_v, does any unvisited vertex become isolated?
            # This is expensive to check fully. Let's just recurse.
            # But we can check: if next_v is the last unvisited vertex, 
            # it must connect to start.
            if len(visited) == _N - 1:
                if start not in adj[next_v]:
                    continue
            
            path.append(next_v)
            visited.add(next_v)
            
            result = dfs(next_v, path, visited)
            if result is not None:
                return result
            
            path.pop()
            visited.remove(next_v)
            
        return None

    return dfs(start, path, visited)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    # Binary search for the minimum K
    # The minimum possible K is _N (a cycle needs N edges)
    # The maximum is total edges
    lo = _N
    hi = len(_EDGE_DATA)
    
    best_cycle = None
    best_K = hi
    
    # Quick check: does a cycle exist at all?
    # For K_24, it definitely does.
    
    while lo <= hi:
        if time.time() > deadline - 5:
            break
            
        mid = (lo + hi) // 2
        # Try to find a cycle with max edge index <= mid
        cycle = _find_hamiltonian_cycle(mid, deadline)
        
        if cycle is not None:
            best_cycle = cycle
            best_K = mid
            hi = mid - 1
        else:
            lo = mid + 1
            
    if best_cycle is None:
        # Fallback to a simple cycle if CSP fails (should not happen for K=276)
        # But let's try a greedy one just in case
        best_cycle = list(range(_N))
        
    return best_cycle
# EVOLVE-BLOCK-END
