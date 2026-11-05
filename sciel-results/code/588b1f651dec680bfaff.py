"""Find Hamilton cycle with earliest edge arrivals in pinned edge order."""
import time
import random

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']
TOTAL_EDGES = len(EDGE_ORDER)

# Build edge index map: (u,v) -> 1-based index in edge_order
def _build_edge_index():
    idx = {}
    for i, (u, v) in enumerate(EDGE_ORDER):
        key = (min(u,v), max(u,v))
        idx[key] = i + 1
    return idx

EDGE_INDEX = _build_edge_index()

def _edge_key(u, v):
    return (min(u,v), max(u,v))

def _find_hamiltonian_cycle(max_edge_pos, deadline):
    """Find a Hamilton cycle using only edges with position <= max_edge_pos.
    Returns list of vertices in cycle order, or None if not found."""
    
    # Build adjacency: for each vertex, list of neighbors connected by edges <= max_edge_pos
    adj = [[] for _ in range(N)]
    for i in range(max_edge_pos):
        u, v = EDGE_ORDER[i]
        adj[u].append(v)
        adj[v].append(u)
    
    # Sort neighbors by edge position for better pruning
    for u in range(N):
        adj[u].sort(key=lambda v: EDGE_INDEX[_edge_key(u, v)])
    
    # Fix vertex 0 as start to break symmetry
    start = 0
    path = [start]
    visited = [False] * N
    visited[start] = True
    
    # Use iterative DFS with backtracking
    # State: (path, visited)
    # We'll use a stack-based approach for efficiency
    
    def backtrack(path, visited):
        if len(path) == N:
            # Check if closing edge exists
            last = path[-1]
            if EDGE_INDEX[_edge_key(last, start)] <= max_edge_pos:
                return path[:]
            return None
        
        current = path[-1]
        for neighbor in adj[current]:
            if not visited[neighbor]:
                # Pruning: check if remaining vertices can still be reached
                # Simple degree check: unvisited vertices must have degree >= 2 in remaining graph
                # (except the last one which needs degree >= 1 to connect back)
                visited[neighbor] = True
                path.append(neighbor)
                
                result = backtrack(path, visited)
                if result is not None:
                    return result
                
                path.pop()
                visited[neighbor] = False
        
        return None
    
    result = backtrack(path, visited)
    return result

def _find_min_prefix(deadline):
    """Binary search for minimum prefix where Hamilton cycle exists."""
    lo, hi = N, TOTAL_EDGES  # At least N edges needed for a cycle
    
    best_cycle = None
    
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        cycle = _find_hamiltonian_cycle(mid, deadline)
        if cycle is not None:
            best_cycle = cycle
            hi = mid - 1
        else:
            lo = mid + 1
    
    return best_cycle

def _cycle_score(cycle):
    """Compute the score (max edge position) of a cycle."""
    max_pos = 0
    n = len(cycle)
    for i in range(n):
        u = cycle[i]
        v = cycle[(i+1) % n]
        pos = EDGE_INDEX[_edge_key(u, v)]
        if pos > max_pos:
            max_pos = pos
    return max_pos

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    # First, try binary search
    cycle = _find_min_prefix(deadline)
    
    if cycle is None:
        # Fallback: use identity cycle
        return list(range(N))
    
    # Verify the cycle
    score = _cycle_score(cycle)
    
    # Try to improve with local search if time remains
    if time.time() < deadline - 5:
        # 2-opt style local improvement
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            best_cycle = cycle
            best_score = score
            for i in range(N):
                for j in range(i+2, N):
                    if j == N-1 and i == 0:
                        continue
                    # Reverse segment i..j
                    new_cycle = cycle[:i] + cycle[i:j+1][::-1] + cycle[j+1:]
                    new_score = _cycle_score(new_cycle)
                    if new_score < best_score:
                        best_score = new_score
                        best_cycle = new_cycle
                        improved = True
            if improved:
                cycle = best_cycle
                score = best_score
    
    return cycle
# EVOLVE-BLOCK-END
