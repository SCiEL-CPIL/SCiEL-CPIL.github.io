import time
import sys
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

n = P['n']
edge_order = P['edge_order']

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
        idx = get_arrival(u, v)
        if idx > max_idx:
            max_idx = idx
    return max_idx + 1  # 1-based

def is_connected(adj, n):
    """Check if the graph is connected using BFS."""
    visited = [False] * n
    queue = deque([0])
    visited[0] = True
    count = 1
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                count += 1
                queue.append(v)
    return count == n

def find_hamiltonian_cycle(max_arrival, time_limit, start_vertex=0):
    """
    Find a Hamiltonian cycle using only edges with arrival <= max_arrival.
    Uses Warnsdorff's heuristic for better performance.
    Returns the cycle as a list of vertices, or None if not found.
    """
    # Build adjacency list
    adj = {i: [] for i in range(n)}
    for idx in range(max_arrival + 1):
        u, v = edge_order[idx]
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degree constraint: every vertex must have degree >= 2
    for v in range(n):
        if len(adj[v]) < 2:
            return None
    
    # Check connectivity
    if not is_connected(adj, n):
        return None
    
    # Sort neighbors by arrival time for better heuristics
    for v in range(n):
        adj[v].sort(key=lambda x: get_arrival(v, x))
    
    deadline = time.time() + time_limit
    result = [None]
    
    # Precompute degrees for Warnsdorff's heuristic
    def warnsdorff_degree(v, visited, path_len):
        """Count unvisited neighbors of v (onward options)."""
        count = 0
        for nb in adj[v]:
            if not visited[nb]:
                count += 1
        return count
    
    def backtrack():
        if time.time() > deadline:
            return False
        if len(path) == n:
            # Check if we can close the cycle
            last = path[-1]
            if get_arrival(last, start_vertex) <= max_arrival:
                result[0] = path[:]
                return True
            return False
        
        current = path[-1]
        remaining = n - len(path)
        
        # Get unvisited neighbors
        candidates = []
        for nb in adj[current]:
            if visited[nb]:
                continue
            # If this is the second-to-last vertex, nb must connect to start
            if remaining == 2:
                if get_arrival(nb, start_vertex) > max_arrival:
                    continue
            candidates.append(nb)
        
        if not candidates:
            return False
        
        # Warnsdorff's heuristic: sort by number of onward options (ascending)
        # Fewer options first - forces us to handle constrained vertices early
        candidates.sort(key=lambda x: warnsdorff_degree(x, visited, len(path)))
        
        for nb in candidates:
            visited[nb] = True
            path.append(nb)
            
            if backtrack():
                return True
            
            path.pop()
            visited[nb] = False
        
        return False
    
    visited = [False] * n
    path = [start_vertex]
    visited[start_vertex] = True
    
    backtrack()
    return result[0]

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin
    
    # Default cycle: 0, 1, 2, ..., 23, 0
    default_cycle = list(range(n))
    best_score = compute_score(default_cycle)
    best_cycle = default_cycle
    
    # Binary search on the threshold
    # Minimum possible: we need at least n edges, so threshold >= n-1 (0-based) = 23
    # Maximum: 275 (last edge index)
    lo = 23
    hi = 275
    found = False
    best_threshold = hi
    
    # First, check if even the full graph has a Hamiltonian cycle (sanity check)
    # The default cycle should work, so it does.
    
    # Binary search for the minimum threshold
    while lo <= hi and time.time() < deadline - 5:
        mid = (lo + hi) // 2
        
        # Try multiple starting vertices for robustness
        cycle = None
        time_per_start = max(0.05, min(0.5, (deadline - 5 - time.time()) / 24.0))
        
        for sv in range(n):
            if time.time() > deadline - 3:
                break
            cycle = find_hamiltonian_cycle(mid, time_per_start, start_vertex=sv)
            if cycle is not None:
                break
        
        if cycle is not None:
            score = compute_score(cycle)
            if score <= mid + 1 and score < best_score:
                best_score = score
                best_cycle = cycle
            found = True
            best_threshold = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    # If binary search found a threshold, try to refine by checking lower values
    if found and best_threshold > 23 and time.time() < deadline - 3:
        # Try a few values below the found threshold to see if we can go lower
        for t in range(max(23, best_threshold - 3), best_threshold):
            if time.time() > deadline - 2:
                break
            cycle = None
            time_per_start = max(0.05, min(0.3, (deadline - 2 - time.time()) / 24.0))
            for sv in range(n):
                if time.time() > deadline - 1:
                    break
                cycle = find_hamiltonian_cycle(t, time_per_start, start_vertex=sv)
                if cycle is not None:
                    break
            if cycle is not None:
                score = compute_score(cycle)
                if score < best_score:
                    best_score = score
                    best_cycle = cycle
    
    return best_cycle
# EVOLVE-BLOCK-END
