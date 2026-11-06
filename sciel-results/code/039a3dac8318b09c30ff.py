"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import time
import random

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

def _build_edge_index():
    n = P['n']
    edge_idx = {}
    for i, (u, v) in enumerate(P['edge_order']):
        edge_idx[(min(u,v), max(u,v))] = i + 1
    return edge_idx

_edge_idx = _build_edge_index()

def _check_connectivity(adj, n):
    """Check if the graph is connected using BFS."""
    if n == 0:
        return True
    visited = [False] * n
    queue = [0]
    visited[0] = True
    count = 1
    while queue:
        curr = queue.pop(0)
        for neighbor in adj[curr]:
            if not visited[neighbor]:
                visited[neighbor] = True
                count += 1
                queue.append(neighbor)
    return count == n

def _find_hamilton_cycle_threshold(K, max_time=5.0):
    """
    Find a Hamilton cycle using only edges with index <= K.
    Returns the cycle as a list of vertices, or None if not found.
    """
    n = P['n']
    start_time = time.time()
    
    # Build adjacency list for edges <= K
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(P['edge_order']):
        if i + 1 > K:
            break
        adj[u].append(v)
        adj[v].append(u)
    
    # Check connectivity
    if not _check_connectivity(adj, n):
        return None
    
    # Check minimum degree >= 2
    for i in range(n):
        if len(adj[i]) < 2:
            return None
    
    # Try multiple starting vertices
    # Sort vertices by degree (try low degree first)
    vertices_by_degree = sorted(range(n), key=lambda x: len(adj[x]))
    
    for start in vertices_by_degree:
        if time.time() - start_time > max_time:
            break
        
        # Sort neighbors by degree (Warnsdorff's rule)
        adj_sorted = [sorted(adj[i], key=lambda x: len(adj[x])) for i in range(n)]
        
        visited = [False] * n
        path = []
        
        visited[start] = True
        path.append(start)
        
        def dfs(current):
            if time.time() - start_time > max_time:
                return False
            
            if len(path) == n:
                # Check if we can return to start
                if start in adj[current]:
                    return True
                return False
            
            # Pruning: Check if any unvisited vertex has degree 0 in remaining graph
            # This is expensive, so only do it periodically
            if len(path) % 4 == 0:
                for v in range(n):
                    if not visited[v]:
                        # Count unvisited neighbors
                        unvisited_neighbors = 0
                        for nb in adj[v]:
                            if not visited[nb]:
                                unvisited_neighbors += 1
                        # Also need to connect to current path eventually
                        # If a vertex has no unvisited neighbors and isn't the last one, it's stuck
                        if unvisited_neighbors == 0 and len(path) < n - 1:
                            return False
            
            # Pruning: If next_v is the last vertex to be visited, it must connect to start
            candidates = adj_sorted[current]
            
            for next_v in candidates:
                if not visited[next_v]:
                    # Pruning: If next_v is the last vertex to be visited, it must connect to start
                    if len(path) == n - 1:
                        if start not in adj[next_v]:
                            continue
                    
                    visited[next_v] = True
                    path.append(next_v)
                    
                    if dfs(next_v):
                        return True
                    
                    path.pop()
                    visited[next_v] = False
            
            return False
        
        if dfs(start):
            # Rotate so start is first
            idx = path.index(start)
            path = path[idx:] + path[:idx]
            return path
    
    return None

def construct():
    n = P['n']
    max_edge_idx = len(P['edge_order'])
    
    # Binary search for the minimum K
    low = 24  # Minimum edges for a cycle
    high = max_edge_idx
    best_cycle = None
    best_K = max_edge_idx + 1
    
    # First, find a valid cycle with a higher K to establish an upper bound
    # Try from the middle first
    for K in range(200, max_edge_idx + 1, 10):
        if time.time() > 50:
            break
        cycle = _find_hamilton_cycle_threshold(K, max_time=2.0)
        if cycle:
            best_cycle = cycle
            best_K = K
            high = K - 1
            break
    
    if best_cycle is None:
        # Try higher K values
        for K in range(max_edge_idx, 200, -10):
            if time.time() > 55:
                break
            cycle = _find_hamilton_cycle_threshold(K, max_time=1.0)
            if cycle:
                best_cycle = cycle
                best_K = K
                break
    
    if best_cycle is None:
        # Fallback: return a random cycle
        rng = random.Random(614)
        cycle = list(range(n))
        rng.shuffle(cycle)
        return cycle
    
    # Now binary search between low and high
    while low <= high and time.time() < 55:
        mid = (low + high) // 2
        cycle = _find_hamilton_cycle_threshold(mid, max_time=3.0)
        if cycle:
            best_cycle = cycle
            best_K = mid
            high = mid - 1
        else:
            low = mid + 1
    
    return best_cycle
# EVOLVE-BLOCK-END
