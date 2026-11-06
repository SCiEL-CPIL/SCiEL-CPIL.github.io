"""Improved component-aware longest path search with better heuristics."""
import time
import random
from collections import defaultdict

# Pinned graph data
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_components(adj):
    """Find all connected components, sorted by size descending."""
    n = P['n']
    visited = [False] * n
    components = []
    
    for i in range(n):
        if not visited[i]:
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for neighbor in adj[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)
            components.append(comp)
    
    # Sort by size descending to process largest first
    components.sort(key=len, reverse=True)
    return components

def _warnsdorff_score(node, seen, adj):
    """Score for Warnsdorff-like ordering: prefer neighbors with fewer unvisited neighbors."""
    unvisited = adj[node] - seen
    return (len(unvisited), node)

def _dfs_path_limited(adj, start, deadline, time_check, max_depth=None):
    """DFS from start with strict time limit and optional depth limit."""
    n = P['n']
    best = [start]
    best_len = 1
    calls = [0]
    
    def dfs(node, path, seen):
        nonlocal best, best_len
        if time_check():
            return
        calls[0] += 1
        if max_depth and len(path) >= max_depth:
            return
        
        if len(path) > best_len:
            best = path[:]
            best_len = len(path)
        
        # Sort neighbors: prefer those with fewer unvisited neighbors (Warnsdorff)
        unvisited = adj[node] - seen
        if not unvisited:
            return
        
        neighbors = sorted(unvisited, key=lambda w: (len(adj[w] - seen), w))
        
        for v in neighbors:
            if time_check():
                return
            path.append(v)
            seen.add(v)
            dfs(v, path, seen)
            path.pop()
            seen.remove(v)
    
    dfs(start, [start], {start})
    return best

def _greedy_extend(adj, path, deadline, time_check):
    """Greedy extension from both ends using Warnsdorff heuristic."""
    extended = path[:]
    seen = set(extended)
    
    # Try extending from the end
    while True:
        current = extended[-1]
        candidates = adj[current] - seen
        if not candidates:
            break
        v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
        extended.append(v)
        seen.add(v)
    
    # Try extending from the beginning
    prepended = []
    while True:
        current = extended[0]
        candidates = adj[current] - seen
        if not candidates:
            break
        v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
        prepended.append(v)
        seen.add(v)
    
    return prepended[::-1] + extended

def _local_improve(adj, path, deadline, time_check):
    """Local improvement: try inserting unvisited vertices into the path."""
    n = P['n']
    best = path[:]
    best_len = len(path)
    seen = set(best)
    
    # Try to insert each unvisited vertex into the path
    for v in range(n):
        if time_check():
            break
        if v in seen:
            continue
        
        # Try inserting v between consecutive pairs in the path
        for i in range(len(best) - 1):
            if time_check():
                break
            a, b = best[i], best[i + 1]
            if v in adj[a] and v in adj[b]:
                new_path = best[:i+1] + [v] + best[i+1:]
                if len(new_path) > best_len:
                    best = new_path
                    best_len = len(best)
                    seen = set(best)
                    break
        
        # Try extending from endpoints
        for endpoint in [best[0], best[-1]]:
            if v in adj[endpoint] and v not in seen:
                if endpoint == best[0]:
                    best = [v] + best
                else:
                    best = best + [v]
                best_len = len(best)
                seen = set(best)
                break
    
    return best

def _random_walk_extend(adj, start, rng, deadline, time_check):
    """Random walk with greedy extension."""
    n = P['n']
    path = [start]
    seen = {start}
    
    while True:
        current = path[-1]
        candidates = adj[current] - seen
        if not candidates:
            break
        v = rng.choice(sorted(candidates))
        path.append(v)
        seen.add(v)
    
    return path

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    def time_check():
        return time.time() >= deadline
    
    adj = _build_adj()
    n = P['n']
    
    best = []
    best_len = 0
    
    # Find connected components (sorted by size descending)
    components = _find_components(adj)
    
    # Process each component
    for comp in components:
        if time_check():
            break
        
        # For each component, try DFS from each vertex with limited time
        for start in comp:
            if time_check():
                break
            
            # DFS from this start with limited exploration
            path = _dfs_path_limited(adj, start, deadline, time_check)
            if len(path) > best_len:
                best = path[:]
                best_len = len(path)
        
        # Also try random walks
        rng = random.Random(hash(tuple(sorted(comp))))
        for _ in range(min(15, len(comp))):
            if time_check():
                break
            start = rng.choice(comp)
            path = _random_walk_extend(adj, start, rng, deadline, time_check)
            # Extend the random walk greedily
            extended = _greedy_extend(adj, path, deadline, time_check)
            if len(extended) > best_len:
                best = extended
                best_len = len(extended)
    
    # Final improvement: local insertion and endpoint extension
    if best and not time_check():
        improved = _local_improve(adj, best, deadline, time_check)
        if len(improved) > best_len:
            best = improved
            best_len = len(improved)
        
        # Try greedy extension one more time
        extended = _greedy_extend(adj, best, deadline, time_check)
        if len(extended) > best_len:
            best = extended
            best_len = len(extended)
    
    # Ensure we have at least one valid path
    if not best:
        best = [0]
    
    return best
