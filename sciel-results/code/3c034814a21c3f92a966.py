"""Find a Hamilton cycle minimizing max edge arrival index."""
import time
import random

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']
TOTAL_EDGES = N * (N - 1) // 2  # 276

# Build edge index lookup: (u,v) -> arrival index (1-based)
edge_idx = {}
for i, (u, v) in enumerate(EDGE_ORDER, 1):
    key = (min(u, v), max(u, v))
    edge_idx[key] = i

def get_edge_index(u, v):
    key = (min(u, v), max(u, v))
    return edge_idx[key]

def cycle_score(cycle):
    """Compute the max edge index for a given Hamilton cycle."""
    max_idx = 0
    for i in range(len(cycle)):
        u = cycle[i]
        v = cycle[(i + 1) % len(cycle)]
        idx = get_edge_index(u, v)
        if idx > max_idx:
            max_idx = idx
    return max_idx

def build_adjacency(max_edge):
    """Build adjacency list with only edges <= max_edge."""
    adj = [[] for _ in range(N)]
    for u in range(N):
        for v in range(u + 1, N):
            idx = get_edge_index(u, v)
            if idx <= max_edge:
                adj[u].append(v)
                adj[v].append(u)
    return adj

def find_cycle_warnsdorff(max_edge, time_limit, start_vertex=0):
    """Find a Hamilton cycle using Warnsdorff's rule heuristic."""
    start_time = time.time()
    deadline = start_time + time_limit
    
    adj = build_adjacency(max_edge)
    
    # Check if all vertices have degree >= 2
    for v in range(N):
        if len(adj[v]) < 2:
            return None
    
    visited = [False] * N
    visited[start_vertex] = True
    path = [start_vertex]
    
    def dfs(current, depth):
        if time.time() > deadline:
            return False
        if depth == N:
            # Check closing edge
            if 0 in adj[current]:
                close_idx = get_edge_index(current, 0)
                if close_idx <= max_edge:
                    return True
            return False
        
        # Warnsdorff's rule: sort neighbors by their degree (fewest options first)
        neighbors = [nxt for nxt in adj[current] if not visited[nxt]]
        if not neighbors:
            return False
        
        # Sort by degree in the remaining graph (approximate with current degree)
        neighbors.sort(key=lambda v: len([n for n in adj[v] if not visited[n]]))
        
        for nxt in neighbors:
            visited[nxt] = True
            path.append(nxt)
            if dfs(nxt, depth + 1):
                return True
            path.pop()
            visited[nxt] = False
        return False
    
    if dfs(start_vertex, 1):
        return list(path)
    return None

def local_search_polish(cycle, max_edge, time_limit):
    """Try to improve a cycle by reducing its max edge index via 2-opt style moves."""
    start_time = time.time()
    deadline = start_time + time_limit
    
    current_cycle = list(cycle)
    current_score = cycle_score(current_cycle)
    
    if current_score <= max_edge:
        return current_cycle
    
    # Try to reduce the max edge by swapping segments
    for _ in range(100):  # Limited iterations
        if time.time() > deadline:
            break
        
        # Find the edge with the highest index
        max_idx = 0
        max_pos = -1
        for i in range(len(current_cycle)):
            u = current_cycle[i]
            v = current_cycle[(i + 1) % len(current_cycle)]
            idx = get_edge_index(u, v)
            if idx > max_idx:
                max_idx = idx
                max_pos = i
        
        # Try to replace this edge with a better one using 2-opt
        i1 = max_pos
        i2 = (i1 + 1) % len(current_cycle)
        
        # Try swapping with another edge
        for j in range(len(current_cycle)):
            if j == i1 or j == i2:
                continue
            j1 = j
            j2 = (j + 1) % len(current_cycle)
            
            # Try reversing segment between i1 and j1
            a, b = current_cycle[i1], current_cycle[i2]
            c, d = current_cycle[j1], current_cycle[j2]
            
            new_idx_ab = get_edge_index(a, c)
            new_idx_bd = get_edge_index(b, d)
            
            if new_idx_ab < max_idx and new_idx_bd < max_idx:
                # Check if this creates a valid cycle
                # Reverse the segment
                new_cycle = current_cycle[:i1+1] + current_cycle[j1:i2:-1] + current_cycle[i2:j1+1]
                # Actually, let's do a proper 2-opt
                # This is complex; for now, just try simple swaps
                
                # Try direct edge swap
                new_cycle = list(current_cycle)
                new_cycle[i1] = c
                new_cycle[j1] = a
                
                # Verify it's still a valid Hamilton cycle
                if len(set(new_cycle)) == N:
                    new_score = cycle_score(new_cycle)
                    if new_score < current_score:
                        current_cycle = new_cycle
                        current_score = new_score
                        break
    
    return current_cycle

def construct():
    start_time = time.time()
    overall_deadline = start_time + 85  # Safety margin before 90s
    
    # Binary search on the maximum edge index
    lo = N  # minimum possible (24 edges needed)
    hi = TOTAL_EDGES
    
    best_cycle = None
    best_max_edge = TOTAL_EDGES + 1
    
    # First, find any valid cycle with all edges
    initial_cycle = find_cycle_warnsdorff(TOTAL_EDGES, 5.0)
    if initial_cycle is not None:
        best_cycle = initial_cycle
        best_max_edge = cycle_score(initial_cycle)
        hi = best_max_edge - 1
    
    # Binary search to find the minimum max_edge
    while lo <= hi and (time.time() - start_time) < 80:
        mid = (lo + hi) // 2
        
        # Allocate time proportional to remaining budget
        remaining = 80 - (time.time() - start_time)
        time_for_search = min(remaining * 0.7, 8.0)  # max 8s per search
        if time_for_search < 1:
            break
        
        # Try multiple starting vertices for better coverage
        found = False
        for start_v in range(N):
            if time.time() - start_time > 80:
                break
            time_per_start = (time_for_search - (time.time() - start_time)) / max(1, N - start_v)
            if time_per_start < 0.1:
                break
            
            cycle = find_cycle_warnsdorff(mid, time_per_start, start_vertex=start_v)
            if cycle is not None:
                score = cycle_score(cycle)
                if score <= mid:
                    best_cycle = cycle
                    best_max_edge = mid
                    hi = mid - 1
                    found = True
                    break
        
        if not found:
            lo = mid + 1
    
    # If we found a cycle, try to polish it
    if best_cycle is not None:
        remaining_time = 85 - (time.time() - start_time)
        if remaining_time > 2:
            polished = local_search_polish(best_cycle, best_max_edge, remaining_time - 1)
            if polished is not None:
                polished_score = cycle_score(polished)
                if polished_score < best_max_edge:
                    best_cycle = polished
                    best_max_edge = polished_score
    
    if best_cycle is None:
        # Fallback: return identity permutation
        return list(range(N))
    
    return best_cycle
# EVOLVE-BLOCK-END
