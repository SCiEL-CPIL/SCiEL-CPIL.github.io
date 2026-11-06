"""Rewritten program for the suite task; stronger search within the same family."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    adj_list = [sorted(adj[i]) for i in range(n)]
    return adj_list

def _greedy_extend(start, adj_list, n, deadline):
    """Greedy extension using Warnsdorff-like heuristic."""
    visited = [False] * n
    visited[start] = True
    path = [start]
    current = start
    
    while True:
        if time.time() > deadline:
            break
        neighbors = [v for v in adj_list[current] if not visited[v]]
        if not neighbors:
            break
        neighbors.sort(key=lambda v: sum(1 for w in adj_list[v] if not visited[w]))
        next_v = neighbors[0]
        visited[next_v] = True
        path.append(next_v)
        current = next_v
    
    return path

def _two_end_extension(path, adj_list, n, deadline):
    """Try to extend a path from both ends."""
    if not path:
        return path
    
    visited = set(path)
    extended = list(path)
    
    changed = True
    while changed:
        changed = False
        if time.time() > deadline:
            break
        end = extended[-1]
        neighbors = [v for v in adj_list[end] if v not in visited]
        if neighbors:
            neighbors.sort(key=lambda v: sum(1 for w in adj_list[v] if w not in visited))
            next_v = neighbors[0]
            extended.append(next_v)
            visited.add(next_v)
            changed = True
    
    changed = True
    while changed:
        changed = False
        if time.time() > deadline:
            break
        start_v = extended[0]
        neighbors = [v for v in adj_list[start_v] if v not in visited]
        if neighbors:
            neighbors.sort(key=lambda v: sum(1 for w in adj_list[v] if w not in visited))
            next_v = neighbors[0]
            extended.insert(0, next_v)
            visited.add(next_v)
            changed = True
    
    return extended

def _backtrack_search(start, adj_list, n, best_len, deadline, max_depth=None):
    """DFS backtracking search from a given start vertex with improved heuristics."""
    visited = [False] * n
    visited[start] = True
    path = [start]
    best_path = [start]
    best_depth = 0
    
    # Precompute degree for sorting
    deg = [len(adj_list[i]) for i in range(n)]
    
    def dfs(current, depth):
        nonlocal best_path, best_depth
        
        if time.time() > deadline:
            return
        
        if depth > best_depth:
            best_depth = depth
            best_path = path[:]
        
        # Pruning: if we can't beat best, return early
        # Upper bound: current depth + number of unvisited vertices
        unvisited_count = n - depth
        if depth + unvisited_count <= best_depth:
            return
        
        neighbors = [v for v in adj_list[current] if not visited[v]]
        if not neighbors:
            return
        
        # Sort by Warnsdorff (fewest unvisited neighbors first)
        neighbors.sort(key=lambda v: sum(1 for w in adj_list[v] if not visited[w]))
        
        for v in neighbors:
            if time.time() > deadline:
                return
            visited[v] = True
            path.append(v)
            dfs(v, depth + 1)
            path.pop()
            visited[v] = False
    
    dfs(start, 1)
    return best_path, best_depth

def _path_from_edge(u, v, adj_list, n, deadline):
    """Build a path starting from edge (u,v) and extending both ways."""
    visited = {u, v}
    path = [u, v]
    
    # Extend right from v
    changed = True
    while changed:
        changed = False
        if time.time() > deadline:
            break
        end = path[-1]
        neighbors = [w for w in adj_list[end] if w not in visited]
        if neighbors:
            neighbors.sort(key=lambda w: sum(1 for x in adj_list[w] if x not in visited))
            path.append(neighbors[0])
            visited.add(neighbors[0])
            changed = True
    
    # Extend left from u
    changed = True
    while changed:
        changed = False
        if time.time() > deadline:
            break
        start_v = path[0]
        neighbors = [w for w in adj_list[start_v] if w not in visited]
        if neighbors:
            neighbors.sort(key=lambda w: sum(1 for x in adj_list[w] if x not in visited))
            path.insert(0, neighbors[0])
            visited.add(neighbors[0])
            changed = True
    
    return path

def _find_best_start(adj_list, n):
    """Find the best starting vertex for greedy extension."""
    # Start from vertices with highest degree first (more options)
    starts = sorted(range(n), key=lambda v: -len(adj_list[v]))
    return starts

def construct():
    n = P['n']
    adj_list = _build_adj(n, P['edges'])
    
    deadline = time.time() + 85.0  # 85s budget, safety margin before 90s
    best_path = []
    best_len = 0
    
    # Phase 1: Greedy extension from each vertex (sorted by degree)
    start_order = _find_best_start(adj_list, n)
    
    for start in start_order:
        if time.time() > deadline:
            break
        path = _greedy_extend(start, adj_list, n, deadline)
        extended = _two_end_extension(path, adj_list, n, deadline)
        if len(extended) > best_len:
            best_len = len(extended)
            best_path = extended[:]
    
    # Phase 2: Try building paths from each edge
    for u in range(n):
        if time.time() > deadline:
            break
        for v in adj_list[u]:
            if u >= v:
                continue
            if time.time() > deadline:
                break
            path = _path_from_edge(u, v, adj_list, n, deadline)
            if len(path) > best_len:
                best_len = len(path)
                best_path = path[:]
    
    # Phase 3: Backtracking search from promising starts
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        # Focus on starts that gave good greedy results
        for start in start_order:
            if time.time() > deadline:
                break
            
            # Quick greedy to estimate potential
            greedy_path = _greedy_extend(start, adj_list, n, deadline)
            if len(greedy_path) >= best_len - 3:
                # Worth trying backtracking
                bt_path, bt_len = _backtrack_search(start, adj_list, n, best_len, deadline)
                if bt_len > best_len:
                    best_len = bt_len
                    best_path = bt_path[:]
    
    # Phase 4: Final two-end extension on best path
    if best_path and time.time() < deadline:
        extended = _two_end_extension(best_path, adj_list, n, deadline)
        if len(extended) > best_len:
            best_len = len(extended)
            best_path = extended[:]
    
    # Fallback
    if not best_path:
        best_path = [0]
    
    return best_path
# EVOLVE-BLOCK-END
