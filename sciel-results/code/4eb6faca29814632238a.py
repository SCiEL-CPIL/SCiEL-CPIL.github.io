"""Find early Hamilton cycles in a fixed edge process on K_24."""
import time
import sys
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']

edge_pos = {}
for i, (u, v) in enumerate(EDGE_ORDER, 1):
    edge_pos[(min(u, v), max(u, v))] = i

def get_edge_pos(u, v):
    return edge_pos[(min(u, v), max(u, v))]

def build_adjacency(threshold):
    adj = [[] for _ in range(N)]
    for (u, v), pos in edge_pos.items():
        if pos <= threshold:
            adj[u].append(v)
            adj[v].append(u)
    for i in range(N):
        adj[i].sort(key=lambda x: get_edge_pos(i, x))
    return adj

def two_core_peel(adj):
    """Return set of vertices in the 2-core of the graph."""
    deg = [len(adj[i]) for i in range(N)]
    in_core = [True] * N
    queue = deque([i for i in range(N) if deg[i] < 2])
    while queue:
        v = queue.popleft()
        if not in_core[v]:
            continue
        in_core[v] = False
        for nb in adj[v]:
            if in_core[nb]:
                deg[nb] -= 1
                if deg[nb] < 2:
                    queue.append(nb)
    return set(i for i in range(N) if in_core[i])

def find_hamilton_cycle_fast(threshold, deadline):
    """
    Fast Hamilton cycle finder using:
    1. 2-core peeling pre-filter
    2. DFS with aggressive degree-based pruning and Warnsdorff ordering
    """
    adj = build_adjacency(threshold)
    
    for i in range(N):
        if len(adj[i]) < 2:
            return None
    
    core = two_core_peel(adj)
    if len(core) < N:
        return None
    
    # Connectivity check
    visited = [False] * N
    stack = [0]
    visited[0] = True
    count = 0
    while stack:
        node = stack.pop()
        count += 1
        for nb in adj[node]:
            if not visited[nb]:
                visited[nb] = True
                stack.append(nb)
    if count < N:
        return None
    
    start = 0
    path = [start]
    used = [False] * N
    used[start] = True
    
    def backtrack():
        if time.time() > deadline:
            return False
        
        if len(path) == N:
            last = path[-1]
            if get_edge_pos(last, start) <= threshold:
                return True
            return False
        
        current = path[-1]
        is_last_step = (len(path) == N - 1)
        
        candidates = []
        for nxt in adj[current]:
            if used[nxt]:
                continue
            if is_last_step and get_edge_pos(nxt, start) > threshold:
                continue
            candidates.append(nxt)
        
        if not candidates:
            return False
        
        # Warnsdorff: sort by number of unvisited neighbors (fewest first)
        def heuristic_score(nxt):
            if is_last_step:
                return 0
            unvisited_nbrs = 0
            for nb in adj[nxt]:
                if not used[nb] and nb != start:
                    unvisited_nbrs += 1
            return unvisited_nbrs
        
        candidates.sort(key=heuristic_score)
        
        for nxt in candidates:
            # Pruning: check if any unvisited vertex would be left with < 2 available neighbors
            if not is_last_step:
                prune = False
                for v in range(N):
                    if used[v] or v == start or v == current or v == nxt:
                        continue
                    avail = 0
                    for nb in adj[v]:
                        if nb == start:
                            continue
                        if used[nb]:
                            avail += 1
                        elif nb != nxt:
                            avail += 1
                    if avail < 2:
                        prune = True
                        break
                if prune:
                    continue
            
            used[nxt] = True
            path.append(nxt)
            
            if backtrack():
                return True
            
            path.pop()
            used[nxt] = False
        
        return False
    
    if backtrack():
        return path
    return None

def score_cycle(cycle):
    max_pos = 0
    for i in range(len(cycle)):
        u = cycle[i]
        v = cycle[(i + 1) % len(cycle)]
        pos = get_edge_pos(u, v)
        if pos > max_pos:
            max_pos = pos
    return max_pos

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    total_edges = len(EDGE_ORDER)
    
    best_cycle = None
    best_score = total_edges + 1
    
    cycle = find_hamilton_cycle_fast(total_edges, deadline)
    if cycle is not None:
        best_cycle = cycle
        best_score = score_cycle(cycle)
    
    low = N
    high = total_edges
    
    while low <= high and time.time() < deadline:
        mid = (low + high) // 2
        cycle = find_hamilton_cycle_fast(mid, deadline)
        if cycle is not None:
            s = score_cycle(cycle)
            if s < best_score:
                best_score = s
                best_cycle = cycle
            high = mid - 1
        else:
            low = mid + 1
    
    if best_cycle is not None and time.time() < deadline:
        for t in range(best_score - 1, max(N - 1, best_score - 30), -1):
            if time.time() > deadline:
                break
            cycle = find_hamilton_cycle_fast(t, deadline)
            if cycle is not None:
                s = score_cycle(cycle)
                if s < best_score:
                    best_score = s
                    best_cycle = cycle
    
    if best_cycle is None:
        return list(range(N))
    
    return best_cycle
# EVOLVE-BLOCK-END
