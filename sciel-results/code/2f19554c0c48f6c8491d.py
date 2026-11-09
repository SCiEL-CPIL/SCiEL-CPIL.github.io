"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _check_solution_fast(n, base_edges, added_edges, max_diameter=4):
    all_edges = list(base_edges) + list(added_edges)
    
    # Check for duplicate edges
    edge_set = set()
    for u, v in all_edges:
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in all_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Check connectedness
    visited = [False] * n
    visited[0] = True
    q = deque([0])
    count = 1
    while q:
        u = q.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                count += 1
                q.append(v)
    if count != n:
        return False
    
    # Check triangle-free
    for u, v in all_edges:
        if adj[u] & adj[v]:
            return False
    
    # Check diameter via BFS
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    if dist[v] > max_diameter:
                        return False
                    q.append(v)
        if -1 in dist:
            return False
    
    return True

def _get_diameter(n, all_edges):
    adj = [set() for _ in range(n)]
    for u, v in all_edges:
        adj[u].add(v)
        adj[v].add(u)
    max_d = 0
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    if dist[v] > max_d:
                        max_d = dist[v]
                    q.append(v)
    return max_d

def _is_valid_add(n, all_edges, u, v):
    if u == v:
        return False
    key = (min(u, v), max(u, v))
    edge_set = set((min(a,b), max(a,b)) for a, b in all_edges)
    if key in edge_set:
        return False
    adj = [set() for _ in range(n)]
    for a, b in all_edges:
        adj[a].add(b)
        adj[b].add(a)
    if adj[u] & adj[v]:
        return False
    return True

def _local_search(n, base_edges, added_edges, max_diameter=4, deadline=None):
    current = [list(e) for e in added_edges]
    
    # Repeatedly try to remove edges
    improved = True
    while improved:
        if deadline and time.time() > deadline:
            break
        improved = False
        for i in range(len(current)):
            if deadline and time.time() > deadline:
                break
            test = current[:i] + current[i+1:]
            if _check_solution_fast(n, base_edges, test, max_diameter):
                current = test
                improved = True
                break
    
    # Try 1-for-1 swaps to enable further removals
    improved = True
    while improved:
        if deadline and time.time() > deadline:
            break
        improved = False
        for i in range(len(current)):
            if deadline and time.time() > deadline:
                break
            without_i = current[:i] + current[i+1:]
            all_no_i = list(base_edges) + without_i
            adj_no_i = [set() for _ in range(n)]
            for a, b in all_no_i:
                adj_no_i[a].add(b)
                adj_no_i[b].add(a)
            edge_set_no_i = set((min(a,b), max(a,b)) for a, b in all_no_i)
            
            found_swap = False
            for u in range(n):
                if deadline and time.time() > deadline:
                    break
                if found_swap:
                    break
                for v in range(u+1, n):
                    if deadline and time.time() > deadline:
                        break
                    key = (min(u,v), max(u,v))
                    if key in edge_set_no_i:
                        continue
                    if adj_no_i[u] & adj_no_i[v]:
                        continue
                    test = without_i + [[u, v]]
                    if _check_solution_fast(n, base_edges, test, max_diameter):
                        # Check if we can then remove something else
                        current = test
                        improved = True
                        found_swap = True
                        break
            if found_swap:
                break
    
    # Final removal pass
    improved = True
    while improved:
        if deadline and time.time() > deadline:
            break
        improved = False
        for i in range(len(current)):
            if deadline and time.time() > deadline:
                break
            test = current[:i] + current[i+1:]
            if _check_solution_fast(n, base_edges, test, max_diameter):
                current = test
                improved = True
                break
    
    return current

def _greedy_construct(n, base_edges, max_diameter=4, deadline=None):
    added = []
    all_edges = list(base_edges)
    
    for _ in range(20):
        if deadline and time.time() > deadline:
            break
        max_d = _get_diameter(n, all_edges)
        if max_d <= max_diameter:
            break
        
        best_edge = None
        best_d = max_d
        
        for u in range(n):
            if deadline and time.time() > deadline:
                break
            for v in range(u+1, n):
                if deadline and time.time() > deadline:
                    break
                if not _is_valid_add(n, all_edges, u, v):
                    continue
                test_edges = all_edges + [[u, v]]
                test_d = _get_diameter(n, test_edges)
                if test_d < best_d:
                    best_d = test_d
                    best_edge = (u, v)
                if best_d <= max_diameter:
                    break
            if best_d <= max_diameter:
                break
        
        if best_edge is None:
            break
        u, v = best_edge
        added.append([u, v])
        all_edges = all_edges + [[u, v]]
    
    return added

def construct():
    start = time.time()
    deadline = start + 70
    n = P['n']
    base_edges = P['edges']
    max_diameter = P['diameter']
    
    best_result = None
    best_count = float('inf')
    
    # Try multiple initial strategies
    strategies = []
    
    # Strategy: two hubs with spokes
    for h1 in range(3, 12):
        for h2 in range(h1+3, 21):
            s = [[h1, h2]]
            # Add spokes to cover uncovered vertices
            for target in range(n):
                if target == h1 or target == h2:
                    continue
                # Check if target is within distance 4 of h1 or h2 on path
                d1 = abs(target - h1)
                d2 = abs(target - h2)
                if d1 > 3 and d2 > 3:
                    # Need a spoke
                    if d1 <= d2:
                        s.append([h1, target])
                    else:
                        s.append([h2, target])
            strategies.append(s)
    
    # Strategy: single hub
    for hub in range(4, 20):
        s = []
        for target in range(n):
            if target == hub:
                continue
            d = abs(target - hub)
            if d > 3:
                s.append([hub, target])
        strategies.append(s)
    
    # Strategy: greedy
    strategies.append(_greedy_construct(n, base_edges, max_diameter, deadline))
    
    # Evaluate strategies
    for strategy in strategies:
        if time.time() > deadline:
            break
        if len(strategy) >= best_count:
            continue
        if not _check_solution_fast(n, base_edges, strategy, max_diameter):
            continue
        result = _local_search(n, base_edges, strategy, max_diameter, deadline)
        if _check_solution_fast(n, base_edges, result, max_diameter):
            if len(result) < best_count:
                best_count = len(result)
                best_result = result
                if best_count <= 5:
                    break
    
    if best_result is None:
        # Fallback
        best_result = [[12, 1], [12, 3], [12, 5], [12, 7], [12, 9], [12, 15], [12, 17], [12, 19], [12, 21], [12, 23]]
    
    return [[int(u), int(v)] for u, v in best_result]
# EVOLVE-BLOCK-END
