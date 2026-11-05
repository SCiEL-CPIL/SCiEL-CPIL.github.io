import random
import time
import sys

sys.setrecursionlimit(10000)

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_INDEX = {}
for i, edge in enumerate(P['edge_order']):
    u, v = edge
    if u > v:
        u, v = v, u
    EDGE_INDEX[(u, v)] = i + 1

def get_edge_index(u, v):
    if u > v:
        u, v = v, u
    return EDGE_INDEX.get((u, v), 10**9)

ADJ_SORTED = [[] for _ in range(N)]
for u in range(N):
    for v in range(N):
        if u != v:
            idx = get_edge_index(u, v)
            ADJ_SORTED[u].append((idx, v))
    ADJ_SORTED[u].sort(key=lambda x: x[0])

GLOBAL_START_TIME = 0
TIME_LIMIT = 55.0

def has_hamilton_cycle(max_idx):
    """
    Checks if there is a Hamiltonian cycle using only edges with index <= max_idx.
    Returns the cycle if found, else None.
    """
    if time.time() - GLOBAL_START_TIME > TIME_LIMIT:
        return None

    adj = [[] for _ in range(N)]
    for u in range(N):
        for idx, v in ADJ_SORTED[u]:
            if idx <= max_idx:
                adj[u].append(v)
            else:
                break
        
        if len(adj[u]) < 2:
            return None

    best_path = []
    found = False
    start = 0
    visited = [False] * N
    path = [start]
    visited[start] = True

    def dfs(u, depth):
        nonlocal found, best_path
        if found or time.time() - GLOBAL_START_TIME > TIME_LIMIT:
            return

        if depth == N:
            if get_edge_index(u, path[0]) <= max_idx:
                best_path = path[:]
                found = True
            return

        # Pruning for last step
        if depth == N - 1:
            # Find the last unvisited vertex
            last_v = -1
            for v in range(N):
                if not visited[v]:
                    last_v = v
                    break
            if last_v != -1:
                if get_edge_index(u, last_v) <= max_idx and get_edge_index(last_v, start) <= max_idx:
                    best_path = path + [last_v]
                    found = True
            return

        for v in adj[u]:
            if not visited[v]:
                # Pruning: If v is not start, and v has only 1 unvisited neighbor (besides u),
                # we might be forced? 
                # Simple check: if v != start, count unvisited neighbors of v.
                # If count == 1, that neighbor MUST be the next step?
                # This is a strong constraint.
                if v != start:
                    unvisited_neighbors = 0
                    next_forced = -1
                    for w in adj[v]:
                        if not visited[w] and w != u:
                            unvisited_neighbors += 1
                            next_forced = w
                    if unvisited_neighbors == 1:
                        # We must go to next_forced after v?
                        # No, we are at u, going to v.
                        # After v, we must go to next_forced.
                        # So the path is ... u -> v -> next_forced ...
                        # This doesn't prevent us from trying v, but it tells us the next step is fixed.
                        # We can just let the recursion handle it, but we can prune if next_forced is not connected to the rest?
                        # Let's skip this complex pruning for now and rely on the base case.
                        pass

                visited[v] = True
                path.append(v)
                dfs(v, depth + 1)
                path.pop()
                visited[v] = False
                if found:
                    return

    dfs(start, 1)
    
    if found and best_path:
        return best_path
    return None

def greedy_construct():
    """
    Fallback greedy construction.
    """
    path = [0]
    visited = {0}
    for _ in range(N - 1):
        u = path[-1]
        best_v = -1
        best_idx = float('inf')
        for v in range(N):
            if v not in visited:
                idx = get_edge_index(u, v)
                if idx < best_idx:
                    best_idx = idx
                    best_v = v
        if best_v == -1:
            return None
        path.append(best_v)
        visited.add(best_v)
    
    # Check closing edge
    if get_edge_index(path[-1], path[0]) < float('inf'):
        return path
    return None

def construct():
    global GLOBAL_START_TIME
    GLOBAL_START_TIME = time.time()
    
    best_cycle = None
    low = N
    high = 276
    
    while low <= high and time.time() - GLOBAL_START_TIME < TIME_LIMIT:
        mid = (low + high) // 2
        cycle = has_hamilton_cycle(mid)
        if cycle:
            best_cycle = cycle
            high = mid - 1
        else:
            low = mid + 1
            
    if best_cycle is None:
        best_cycle = greedy_construct()
        
    if best_cycle is None:
        return list(range(N))
        
    return best_cycle
# EVOLVE-BLOCK-END
