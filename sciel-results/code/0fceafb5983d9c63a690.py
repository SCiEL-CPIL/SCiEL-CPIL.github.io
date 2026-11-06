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
    # Build adjacency list and bitmask for G_k
    adj_mask = [0] * n
    for i in range(k):
        u, v = edge_order[i]
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    # Quick checks
    for i in range(n):
        if adj_mask[i].bit_count() < 2:
            return None
            
    # Check connectivity using BFS
    visited = [False] * n
    queue = deque([0])
    visited[0] = True
    count = 1
    while queue:
        u = queue.popleft()
        mask = adj_mask[u]
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            if not visited[v]:
                visited[v] = True
                count += 1
                queue.append(v)
            mask ^= lsb
    if count < n:
        return None

    deadline = time.time() + time_limit
    visited_mask = 1 << 0
    path = [0]
    
    def backtrack():
        if time.time() > deadline:
            return None
        
        if len(path) == n:
            last = path[-1]
            if (adj_mask[last] & 1) != 0: # Check edge to 0
                return path[:]
            return None
        
        current = path[-1]
        candidates_mask = adj_mask[current] & ~visited_mask
        
        if candidates_mask == 0:
            return None
            
        # Warnsdorff's Rule: pick the candidate with the fewest unvisited neighbors
        best_candidate = -1
        min_degree = n + 1
        
        temp_mask = candidates_mask
        while temp_mask:
            lsb = temp_mask & -temp_mask
            v = lsb.bit_length() - 1
            # Count unvisited neighbors of v
            deg = (adj_mask[v] & ~visited_mask).bit_count()
            
            if deg < min_degree:
                min_degree = deg
                best_candidate = v
                if deg == 1: # Can't get better than 1 (forced move)
                    break
            temp_mask ^= lsb
            
        if best_candidate == -1:
            return None
            
        visited_mask |= (1 << best_candidate)
        path.append(best_candidate)
        
        result = backtrack()
        if result is not None:
            return result
        
        path.pop()
        visited_mask ^= (1 << best_candidate)
        
        return None

    return backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    # Default cycle
    default_cycle = list(range(n))
    best_score = compute_score(default_cycle)
    best_cycle = default_cycle
    
    # Linear scan for minimum k
    for k in range(24, 277):
        if time.time() > deadline - 5:
            break
            
        # Allocate time based on k
        t_limit = 1.0
        
        cycle = find_hamiltonian_cycle(k, t_limit)
        if cycle is not None:
            score = compute_score(cycle)
            return cycle
            
    return best_cycle

# EVOLVE-BLOCK-END
