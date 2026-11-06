"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from itertools import combinations

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']
NUM_EDGES = len(EDGE_ORDER)

# Build position lookup: edge (u,v) -> 1-based position in edge_order
edge_pos = {}
for i, (u, v) in enumerate(EDGE_ORDER):
    key = (min(u, v), max(u, v))
    edge_pos[key] = i + 1

def get_edge_pos(u, v):
    key = (min(u, v), max(u, v))
    return edge_pos.get(key, float('inf'))

def has_hamilton_cycle_fast(adj, n, time_limit=10):
    """Fast Hamilton cycle check with Warnsdorff's rule and connectivity pruning."""
    start_time = time.time()
    
    deg = [len(adj[i]) for i in range(n)]
    for i in range(n):
        if deg[i] < 2:
            return False
    
    # Connectivity check via BFS
    visited_bfs = [False] * n
    queue = [0]
    visited_bfs[0] = True
    count = 1
    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if not visited_bfs[v]:
                visited_bfs[v] = True
                count += 1
                queue.append(v)
    if count < n:
        return False
    
    def backtrack(path, visited):
        if time.time() - start_time > time_limit:
            return False
        if len(path) == n:
            return 0 in adj[path[-1]]
        
        curr = path[-1]
        remaining = n - len(path)
        
        # Warnsdorff's rule: prefer neighbors with fewest unvisited neighbors
        candidates = []
        for nb in adj[curr]:
            if not visited[nb]:
                unvisited_count = sum(1 for x in adj[nb] if not visited[x])
                candidates.append((unvisited_count, nb))
        candidates.sort()
        
        for _, nb in candidates:
            # Pruning: if not last step, nb must have unvisited neighbor
            if remaining > 1:
                if not any(not visited[x] for x in adj[nb]):
                    continue
            
            visited[nb] = True
            path.append(nb)
            if backtrack(path, visited):
                return True
            path.pop()
            visited[nb] = False
        
        return False
    
    visited = [False] * n
    visited[0] = True
    
    for nb in sorted(adj[0], key=lambda x: deg[x]):
        visited[nb] = True
        path = [0, nb]
        if backtrack(path, visited):
            return True
        path.pop()
        visited[nb] = False
    
    return False

def extract_hamilton_cycle_fast(adj, n, time_limit=20):
    """Extract Hamilton cycle using Warnsdorff's rule."""
    start_time = time.time()
    
    deg = [len(adj[i]) for i in range(n)]
    
    def backtrack(path, visited):
        if time.time() - start_time > time_limit:
            return None
        if len(path) == n:
            if 0 in adj[path[-1]]:
                return list(path)
            return None
        
        curr = path[-1]
        
        candidates = []
        for nb in adj[curr]:
            if not visited[nb]:
                unvisited_count = sum(1 for x in adj[nb] if not visited[x])
                candidates.append((unvisited_count, nb))
        candidates.sort()
        
        for _, nb in candidates:
            if len(path) < n - 1:
                if not any(not visited[x] for x in adj[nb]):
                    continue
            
            visited[nb] = True
            path.append(nb)
            result = backtrack(path, visited)
            if result is not None:
                return result
            path.pop()
            visited[nb] = False
        
        return None
    
    visited = [False] * n
    visited[0] = True
    
    for nb in sorted(adj[0], key=lambda x: deg[x]):
        visited[nb] = True
        path = [0, nb]
        result = backtrack(path, visited)
        if result is not None:
            return result
        path.pop()
        visited[nb] = False
    
    return None

def cycle_max_pos(cycle):
    """Compute the max edge position in a cycle."""
    max_pos = 0
    n = len(cycle)
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        pos = get_edge_pos(u, v)
        max_pos = max(max_pos, pos)
    return max_pos

def two_opt_improve(cycle, edge_order, n, time_limit=30):
    """Apply 2-opt swaps to reduce the max edge position."""
    start_time = time.time()
    current_cycle = list(cycle)
    current_max = cycle_max_pos(current_cycle)
    improved = True
    
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i in range(n - 1):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 2, n):
                if j == n - 1 and i == 0:
                    continue  # This would be the same as reversing
                # Try reversing segment i+1..j
                new_cycle = current_cycle[:i+1] + current_cycle[i+1:j+1][::-1] + current_cycle[j+1:]
                new_max = cycle_max_pos(new_cycle)
                if new_max < current_max:
                    current_cycle = new_cycle
                    current_max = new_max
                    improved = True
                    break
            if improved:
                break
    
    return current_cycle, current_max

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin
    
    n = N
    edge_order = EDGE_ORDER
    
    # Strategy: Try thresholds from low to high, find first cycle, then 2-opt improve
    # Start with a reasonable lower bound
    best_cycle = None
    best_max = float('inf')
    
    # Try several thresholds, starting from a low estimate
    # The minimum possible is n (24), but realistically higher
    thresholds_to_try = []
    
    # Binary search for a good starting point
    lo, hi = n, len(edge_order)
    best_k = hi
    while lo <= hi and time.time() - start_time < deadline - 30:
        mid = (lo + hi) // 2
        adj = [set() for _ in range(n)]
        for i in range(mid):
            u, v = edge_order[i]
            adj[u].add(v)
            adj[v].add(u)
        
        if has_hamilton_cycle_fast(adj, n, time_limit=min(5, deadline - 30 - (time.time() - start_time))):
            best_k = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    # Now try thresholds around best_k, from low to high
    search_start = max(n, best_k - 10)
    search_end = min(len(edge_order), best_k + 20)
    
    for k in range(search_start, search_end + 1):
        if time.time() - start_time > deadline - 15:
            break
        
        adj = [set() for _ in range(n)]
        for i in range(k):
            u, v = edge_order[i]
            adj[u].add(v)
            adj[v].add(u)
        
        cycle = extract_hamilton_cycle_fast(adj, n, time_limit=min(5, deadline - 15 - (time.time() - start_time)))
        if cycle is not None:
            cur_max = cycle_max_pos(cycle)
            if cur_max < best_max:
                best_max = cur_max
                best_cycle = cycle
    
    # If we found a cycle, try to improve with 2-opt
    if best_cycle is not None and time.time() - start_time < deadline - 5:
        improved_cycle, improved_max = two_opt_improve(best_cycle, edge_order, n, time_limit=min(10, deadline - 5 - (time.time() - start_time)))
        if improved_max < best_max:
            best_cycle = improved_cycle
            best_max = improved_max
    
    # Fallback: use all edges
    if best_cycle is None and time.time() - start_time < deadline - 5:
        adj = [set() for _ in range(n)]
        for u, v in edge_order:
            adj[u].add(v)
            adj[v].add(u)
        cycle = extract_hamilton_cycle_fast(adj, n, time_limit=min(10, deadline - 5 - (time.time() - start_time)))
        if cycle is not None:
            best_cycle = cycle
    
    if best_cycle is not None:
        return best_cycle
    
    return list(range(n))

# EVOLVE-BLOCK-END
