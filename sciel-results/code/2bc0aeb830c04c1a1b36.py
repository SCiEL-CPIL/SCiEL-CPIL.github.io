"""Find longest simple path in pinned G(100, 1.8/100) graph."""
import time
import random

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_components(adj, n):
    """Find connected components, return list of (component_size, [vertices])."""
    visited = [False] * n
    comps = []
    for start in range(n):
        if not visited[start]:
            stack = [start]
            visited[start] = True
            comp = []
            while stack:
                v = stack.pop()
                comp.append(v)
                for nb in adj[v]:
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)
            comps.append((len(comp), comp))
    comps.sort(key=lambda x: -x[0])
    return comps

def _greedy_path(adj, start, n, time_limit=None):
    """Greedy path: always go to neighbor with fewest unvisited neighbors (dead-end avoidance)."""
    path = [start]
    seen = {start}
    deadline = time.time() + time_limit if time_limit else None
    while True:
        cur = path[-1]
        candidates = [v for v in adj[cur] if v not in seen]
        if not candidates:
            break
        if deadline and time.time() > deadline:
            break
        # Pick neighbor with minimum degree to unvisited (prefer dead-ends)
        best = min(candidates, key=lambda v: (len(adj[v] - seen), v))
        path.append(best)
        seen.add(best)
    return path

def _dfs_longest_path(adj, start, n, max_depth, time_limit):
    """DFS with backtracking to find longest path from start, bounded by max_depth and time."""
    deadline = time.time() + time_limit
    best_path = [start]
    best_len = 1
    
    def dfs(path, seen):
        nonlocal best_path, best_len
        if time.time() > deadline:
            return
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
        if len(path) >= max_depth:
            return
        cur = path[-1]
        candidates = [v for v in adj[cur] if v not in seen]
        if not candidates:
            return
        # Sort by degree (ascending) to try dead-ends first
        candidates.sort(key=lambda v: len(adj[v]))
        for v in candidates:
            if time.time() > deadline:
                return
            path.append(v)
            seen.add(v)
            dfs(path, seen)
            path.pop()
            seen.discard(v)
    
    dfs([start], {start})
    return best_path

def _try_extend_path(adj, path, n, time_limit):
    """Try to extend a path by trying to insert vertices or extend ends."""
    deadline = time.time() + time_limit
    best = list(path)
    best_len = len(best)
    seen = set(best)
    
    # Try extending from both ends
    for end_idx in [0, 1]:  # 0=front, 1=back
        while time.time() < deadline:
            cur = best[0] if end_idx == 0 else best[-1]
            candidates = [v for v in adj[cur] if v not in seen]
            if not candidates:
                break
            best_v = min(candidates, key=lambda v: len(adj[v]))
            if end_idx == 0:
                best.insert(0, best_v)
            else:
                best.append(best_v)
            seen.add(best_v)
    
    # Try inserting vertices between consecutive pairs
    i = 0
    while i < len(best) - 1 and time.time() < deadline:
        a, b = best[i], best[i+1]
        # Find a vertex x such that a-x and x-b are edges, x not in seen
        candidates = [v for v in adj[a] if v in adj[b] and v not in seen and v != a and v != b]
        if candidates:
            x = min(candidates, key=lambda v: len(adj[v]))
            best.insert(i+1, x)
            seen.add(x)
            # Don't increment i, check new position too
        else:
            i += 1
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin for 90s limit
    n = P['n']
    adj = _build_adj(adj_n) if False else _build_adj()
    
    # Find components
    comps = _find_components(adj, n)
    
    # The longest path must be within one component
    # Focus on the largest component
    if not comps or comps[0][0] == 0:
        return [0]
    
    best_path = []
    
    # Strategy 1: Greedy from every vertex in largest component
    largest_comp = comps[0][1]
    for start in largest_comp:
        if time.time() > deadline:
            break
        path = _greedy_path(adj, start, n)
        if len(path) > len(best_path):
            best_path = path
    
    # Strategy 2: Try to extend the best path found
    if len(best_path) > 0:
        extended = _try_extend_path(adj, best_path, n, 1.0)
        if len(extended) > len(best_path):
            best_path = extended
    
    # Strategy 3: DFS with backtracking from promising starts in largest component
    # Only if we have time remaining
    remaining = deadline - time.time()
    if remaining > 2.0 and len(largest_comp) <= 60:
        # Try DFS from vertices with low degree (likely endpoints of long paths)
        comp_set = set(largest_comp)
        # Sort by degree within component
        def deg_in_comp(v):
            return len([u for u in adj[v] if u in comp_set])
        
        starts_sorted = sorted(largest_comp, key=deg_in_comp)
        
        # Try DFS from a few promising starts
        for start in starts_sorted[:10]:
            if time.time() > deadline:
                break
            remaining_for_dfs = min(3.0, deadline - time.time() - 1.0)
            if remaining_for_dfs <= 0:
                break
            # Limit depth to component size
            max_d = min(len(largest_comp), 50)
            dfs_path = _dfs_longest_path(adj, start, n, max_d, remaining_for_dfs)
            if len(dfs_path) > len(best_path):
                best_path = dfs_path
    
    # Strategy 4: More greedy variants with different tie-breaking
    if time.time() < deadline - 1.0:
        for start in largest_comp:
            if time.time() > deadline:
                break
            # Variant: prefer high-degree neighbors first
            path = [start]
            seen = {start}
            while True:
                cur = path[-1]
                candidates = [v for v in adj[cur] if v not in seen]
                if not candidates:
                    break
                # Prefer high degree (more options)
                best_v = max(candidates, key=lambda v: (len(adj[v] - seen), -v))
                path.append(best_v)
                seen.add(best_v)
            if len(path) > len(best_path):
                best_path = path
    
    # Strategy 5: Try all pairs as potential endpoints, build path between them
    if time.time() < deadline - 2.0 and len(largest_comp) <= 40:
        # For small components, try more exhaustive search
        comp_set = set(largest_comp)
        for i, a in enumerate(largest_comp):
            if time.time() > deadline:
                break
            for b in largest_comp[i+1:]:
                if time.time() > deadline:
                    break
                # Try to find a path from a to b
                # Simple BFS-like: try to extend from a toward b
                path = [a]
                seen = {a}
                target = b
                while path[-1] != target and len(path) < len(largest_comp):
                    cur = path[-1]
                    candidates = [v for v in adj[cur] if v not in seen]
                    if not candidates:
                        break
                    # Prefer neighbors that are closer to target (heuristic)
                    # Just pick one with fewest options
                    best_v = min(candidates, key=lambda v: len(adj[v] - seen))
                    path.append(best_v)
                    seen.add(best_v)
                if path[-1] == target and len(path) > len(best_path):
                    best_path = path
    
    # Final: ensure we return a valid non-empty path
    if not best_path:
        # Fallback: return a single vertex or any edge
        if P['edges']:
            best_path = [P['edges'][0][0], P['edges'][0][1]]
        else:
            best_path = [0]
    
    return best_path

# Fix the _build_adj call - I had a bug in the code above
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin for 90s limit
    n = P['n']
    adj = _build_adj()
    
    # Find components
    comps = _find_components(adj, n)
    
    # The longest path must be within one component
    if not comps or comps[0][0] == 0:
        return [0]
    
    best_path = []
    
    # Strategy 1: Greedy from every vertex in largest component
    largest_comp = comps[0][1]
    for start in largest_comp:
        if time.time() > deadline:
            break
        path = _greedy_path(adj, start, n)
        if len(path) > len(best_path):
            best_path = path
    
    # Strategy 2: Try to extend the best path found
    if len(best_path) > 0:
        extended = _try_extend_path(adj, best_path, n, 1.0)
        if len(extended) > len(best_path):
            best_path = extended
    
    # Strategy 3: DFS with backtracking from promising starts in largest component
    remaining = deadline - time.time()
    if remaining > 2.0 and len(largest_comp) <= 60:
        comp_set = set(largest_comp)
        def deg_in_comp(v):
            return len([u for u in adj[v] if u in comp_set])
        
        starts_sorted = sorted(largest_comp, key=deg_in_comp)
        
        for start in starts_sorted[:10]:
            if time.time() > deadline:
                break
            remaining_for_dfs = min(3.0, deadline - time.time() - 1.0)
            if remaining_for_dfs <= 0:
                break
            max_d = min(len(largest_comp), 50)
            dfs_path = _dfs_longest_path(adj, start, n, max_d, remaining_for_dfs)
            if len(dfs_path) > len(best_path):
                best_path = dfs_path
    
    # Strategy 4: More greedy variants
    if time.time() < deadline - 1.0:
        for start in largest_comp:
            if time.time() > deadline:
                break
            path = [start]
            seen = {start}
            while True:
                cur = path[-1]
                candidates = [v for v in adj[cur] if v not in seen]
                if not candidates:
                    break
                best_v = max(candidates, key=lambda v: (len(adj[v] - seen), -v))
                path.append(best_v)
                seen.add(best_v)
            if len(path) > len(best_path):
                best_path = path
    
    # Strategy 5: Try all pairs as potential endpoints for small components
    if time.time() < deadline - 2.0 and len(largest_comp) <= 40:
        for i, a in enumerate(largest_comp):
            if time.time() > deadline:
                break
            for b in largest_comp[i+1:]:
                if time.time() > deadline:
                    break
                path = [a]
                seen = {a}
                target = b
                while path[-1] != target and len(path) < len(largest_comp):
                    cur = path[-1]
                    candidates = [v for v in adj[cur] if v not in seen]
                    if not candidates:
                        break
                    best_v = min(candidates, key=lambda v: len(adj[v] - seen))
                    path.append(best_v)
                    seen.add(best_v)
                if path[-1] == target and len(path) > len(best_path):
                    best_path = path
    
    if not best_path:
        if P['edges']:
            best_path = [P['edges'][0][0], P['edges'][0][1]]
        else:
            best_path = [0]
    
    return best_path
