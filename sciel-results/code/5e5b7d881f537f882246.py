"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
M = len(P['edge_order'])

def build_adjacency(k):
    """Build adjacency list for first k edges."""
    adj = [set() for _ in range(N)]
    for i in range(k):
        a, b = P['edge_order'][i]
        adj[a].add(b)
        adj[b].add(a)
    return adj

def find_hamilton_cycle_fast(adj, start=0, deadline=None):
    """Find a Hamilton cycle using backtracking with strong pruning."""
    n = N
    visited = [False] * n
    path = [start]
    visited[start] = True
    adj_list = [list(s) for s in adj]
    
    # Precompute degree for pruning
    degrees = [len(adj_list[i]) for i in range(n)]
    
    sys.setrecursionlimit(10000)
    
    def backtrack():
        if deadline and time.time() > deadline:
            return False
        if len(path) == n:
            if start in adj[path[-1]]:
                return True
            return False
        
        last = path[-1]
        # Try neighbors in order of lowest degree first (MRV heuristic)
        neighbors = [nb for nb in adj_list[last] if not visited[nb]]
        if not neighbors:
            return False
        
        # If we're at position n-1, only one vertex remains
        if len(path) == n - 1:
            unvisited = [v for v in range(n) if not visited[v]]
            if len(unvisited) == 1:
                last_unvisited = unvisited[0]
                if last_unvisited not in adj_list[last]:
                    return False
                if start not in adj_list[last_unvisited]:
                    return False
                visited[last_unvisited] = True
                path.append(last_unvisited)
                if start in adj_list[last_unvisited]:
                    return True
                path.pop()
                visited[last_unvisited] = False
                return False
        
        # Sort by degree of unvisited neighbors (prefer low-degree)
        def degree_key(nb):
            unvisited_count = sum(1 for x in adj_list[nb] if not visited[x])
            return unvisited_count
        
        neighbors.sort(key=degree_key)
        
        for neighbor in neighbors:
            # Prune: check if any unvisited vertex would become isolated
            if len(path) < n - 2:
                can_proceed = True
                for v in range(n):
                    if not visited[v] and v != neighbor:
                        # Count unvisited neighbors of v
                        unvis_nb = sum(1 for x in adj_list[v] if not visited[x])
                        if unvis_nb == 0:
                            can_proceed = False
                            break
                if not can_proceed:
                    continue
            
            visited[neighbor] = True
            path.append(neighbor)
            if backtrack():
                return True
            path.pop()
            visited[neighbor] = False
        
        return False
    
    if backtrack():
        return path
    return None

def find_hamilton_cycle(adj, start=0, deadline=None):
    """Find a Hamilton cycle in the given adjacency structure using backtracking."""
    n = N
    visited = [False] * n
    path = [start]
    visited[start] = True
    adj_list = [list(s) for s in adj]
    
    # Sort neighbors for deterministic ordering
    for i in range(n):
        adj_list[i].sort()
    
    sys.setrecursionlimit(10000)
    
    def backtrack():
        if deadline and time.time() > deadline:
            return False
        if len(path) == n:
            # Check if we can close the cycle
            if start in adj[path[-1]]:
                return True
            return False
        
        last = path[-1]
        for neighbor in adj_list[last]:
            if not visited[neighbor]:
                # Pruning: if this is the second-to-last vertex, 
                # the last unvisited vertex must connect to start
                if len(path) == n - 1:
                    # Find the only unvisited vertex
                    unvisited = [v for v in range(n) if not visited[v]]
                    if len(unvisited) == 1:
                        last_unvisited = unvisited[0]
                        if last_unvisited != neighbor:
                            continue
                        if start not in adj[last_unvisited]:
                            continue
                
                visited[neighbor] = True
                path.append(neighbor)
                if backtrack():
                    return True
                path.pop()
                visited[neighbor] = False
        return False
    
    if backtrack():
        return path
    return None

def get_score(cycle):
    """Compute the score: largest one-based arrival index of edges in the cycle."""
    edge_positions = {}
    for i, (a, b) in enumerate(P['edge_order'], 1):
        key = (min(a, b), max(a, b))
        edge_positions[key] = i
    
    max_pos = 0
    for i in range(len(cycle)):
        a = cycle[i]
        b = cycle[(i + 1) % len(cycle)]
        key = (min(a, b), max(a, b))
        pos = edge_positions.get(key, 0)
        if pos > max_pos:
            max_pos = pos
    return max_pos

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    # Binary search for the minimum k such that a Hamilton cycle exists in first k edges
    lo, hi = N - 1, M  # At least N-1 edges needed for a cycle (actually N edges)
    lo = N  # A cycle needs at least N edges
    
    best_cycle = list(range(N))  # Fallback: identity permutation
    best_score = get_score(best_cycle)
    
    # First, find any feasible k by binary search
    # We need to find the smallest k where a Hamilton cycle exists
    
    # Check if full graph has a Hamilton cycle (it should, it's complete)
    adj_full = build_adjacency(M)
    cycle_full = find_hamilton_cycle(adj_full, deadline=deadline)
    if cycle_full:
        best_cycle = cycle_full
        best_score = get_score(best_cycle)
    
    # Binary search for minimum k
    lo, hi = N, M
    while lo < hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        adj_mid = build_adjacency(mid)
        cycle_mid = find_hamilton_cycle(adj_mid, deadline=deadline)
        if cycle_mid:
            score_mid = get_score(cycle_mid)
            if score_mid < best_score:
                best_score = score_mid
                best_cycle = cycle_mid
            hi = mid
        else:
            lo = mid + 1
    
    # Try to improve: check if lo gives a better cycle
    if time.time() < deadline:
        adj_lo = build_adjacency(lo)
        cycle_lo = find_hamilton_cycle(adj_lo, deadline=deadline)
        if cycle_lo:
            score_lo = get_score(cycle_lo)
            if score_lo < best_score:
                best_score = score_lo
                best_cycle = cycle_lo
    
    return list(best_cycle)
# EVOLVE-BLOCK-END
