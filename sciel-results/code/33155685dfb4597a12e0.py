import time
import sys
import random
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

def find_hamiltonian_cycle_k(K, time_limit, start_vertex=0):
    """
    Check if a Hamiltonian cycle exists using only edges with arrival <= K.
    Uses backtracking with forced edge propagation and degree pruning.
    Returns the cycle if found, None otherwise.
    """
    # Build adjacency list for edges with arrival <= K
    adj = [[] for _ in range(n)]
    adj_set = [set() for _ in range(n)]
    for idx in range(K + 1):
        u, v = edge_order[idx]
        if v not in adj_set[u]:
            adj[u].append(v)
            adj_set[u].add(v)
        if u not in adj_set[v]:
            adj[v].append(u)
            adj_set[v].add(u)
    
    # Check degree >= 2 for all vertices
    for i in range(n):
        if len(adj[i]) < 2:
            return None

    deadline = time.time() + time_limit
    path = [start_vertex]
    visited = [False] * n
    visited[start_vertex] = True
    
    # Track remaining degree (how many edges each vertex still needs in the cycle)
    # Each vertex needs exactly 2 edges in the Hamiltonian cycle
    remaining_deg = [2] * n
    remaining_deg[start_vertex] = 1  # start vertex will have one edge used when we close the cycle
    
    # For the last vertex to connect back to start
    if get_arrival(start_vertex, start_vertex) > K:  # self-loop, shouldn't happen
        pass
    
    def is_connected(unvisited_set, endpoint):
        """Check if all unvisited vertices are connected to endpoint through unvisited vertices."""
        if not unvisited_set:
            return True
        # BFS/DFS from endpoint through unvisited vertices
        visited_bfs = set()
        queue = deque([endpoint])
        visited_bfs.add(endpoint)
        while queue:
            node = queue.popleft()
            for nb in adj[node]:
                if nb in unvisited_set and nb not in visited_bfs:
                    visited_bfs.add(nb)
                    queue.append(nb)
        return len(visited_bfs) == len(unvisited_set) + 1  # +1 for endpoint itself if it's unvisited
    
    def backtrack():
        if time.time() > deadline:
            return None
        if len(path) == n:
            last = path[-1]
            # Check if last connects back to start with arrival <= K
            if get_arrival(last, start_vertex) <= K:
                return path[:]
            return None
        
        current = path[-1]
        
        # Get unvisited neighbors of current
        candidates = [nb for nb in adj[current] if not visited[nb]]
        
        # If no candidates, dead end
        if not candidates:
            return None
        
        # Sort candidates: prefer those with fewer remaining options (MRV heuristic)
        # A vertex with remaining_deg == 2 and only 2 available neighbors is forced
        def sort_key(nb):
            # Count available neighbors for nb (unvisited + current)
            avail = 0
            for x in adj[nb]:
                if not visited[x] or x == current:
                    avail += 1
            return (avail, nb)
        
        candidates.sort(key=sort_key)
        
        for nb in candidates:
            if visited[nb]:
                continue
            
            # Pruning: If nb is the last unvisited vertex, it must connect to start
            if len(path) == n - 1:
                if get_arrival(nb, start_vertex) > K:
                    continue
            
            # Pruning: Check if choosing nb would leave any unvisited vertex isolated
            # from the path endpoint
            if len(path) < n - 1:
                unvisited_after = set(range(n)) - set(path) - {nb}
                if unvisited_after:
                    # Check connectivity from nb to all unvisited vertices
                    # Simple check: nb must be able to reach all unvisited through unvisited+nb
                    # This is expensive, so do a lighter check
                    # Each unvisited vertex must have at least one neighbor in unvisited_after ∪ {nb}
                    for v in unvisited_after:
                        has_neighbor = False
                        for x in adj[v]:
                            if x == nb or (not visited[x] and x not in path):
                                has_neighbor = True
                                break
                        if not has_neighbor:
                            break
                    else:
                        pass  # All good
                    if not has_neighbor:
                        continue
            
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
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Start with the default cycle 0-1-2-...-23-0
    default_cycle = list(range(n))
    best_score = compute_score(default_cycle)
    best_cycle = default_cycle
    
    # Binary search for the minimum threshold K (0-based arrival index)
    # The score is K+1.
    # Min possible K is 23 (24 edges). Max is 275.
    lo = 23
    hi = 275
    
    # We will scan from low to high.
    # If we find a cycle for K, that's our answer.
    # To be safe, we might miss it if DFS fails due to time limit, so we should
    # try multiple starts or restarts for each K.
    
    for K in range(23, 276):
        if time.time() > deadline - 2:
            break
        
        # If we already have a better score (lower K), break
        if K + 1 >= best_score:
            break
        
        remaining_time = deadline - 2 - time.time()
        if remaining_time < 0.5:
            break
        
        # Allocate time for this K
        # Give more time to smaller K values where the search space is more constrained
        # but also where failure is more likely
        if K < 50:
            t_limit = min(3.0, remaining_time * 0.3)
        elif K < 100:
            t_limit = min(2.0, remaining_time * 0.25)
        else:
            t_limit = min(1.5, remaining_time * 0.2)
        
        # Try multiple starting vertices
        found = False
        for sv in range(n):
            if time.time() > deadline - 1:
                break
            result = find_hamiltonian_cycle_k(K, t_limit, start_vertex=sv)
            if result is not None:
                score = compute_score(result)
                if score < best_score:
                    best_score = score
                    best_cycle = result
                found = True
                break
        
        if found:
            # Found a cycle at this K. Since we scan upwards, this is the minimum.
            break
    
    return best_cycle

# EVOLVE-BLOCK-END
