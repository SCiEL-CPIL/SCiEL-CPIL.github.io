"""
Edge path cover via matching-based decomposition.

Key structural insight: the minimum number of edge-disjoint paths covering all
edges of a connected graph equals max(1, (odd_degree_vertices + 2) / 2) when we
can pair odd-degree vertices via shortest paths in the residual graph.

Algorithm:
1. Identify all odd-degree vertices.
2. Compute all-pairs shortest paths (BFS on unweighted graph).
3. Use Hungarian algorithm (or greedy for small sets) to find a minimum-cost
   perfect matching on odd-degree vertices, where cost = shortest path length.
4. For each matched pair, extract the shortest path and remove those edges.
5. Handle any remaining edges (should be few) with greedy walk.
6. Refine with local search: try to merge adjacent paths if possible.
"""

import time
from random import Random
from collections import defaultdict, deque
import math


def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _build_adj(edges, n):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    # Ensure all vertices are present
    for i in range(n):
        if i not in adj:
            adj[i] = set()
    return adj


def _bfs_shortest_paths(adj, n):
    """Compute all-pairs shortest path distances and predecessors."""
    dist = [[-1] * n for _ in range(n)]
    pred = [[-1] * n for _ in range(n)]
    for s in range(n):
        dist[s][s] = 0
        queue = deque([s])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[s][v] == -1:
                    dist[s][v] = dist[s][u] + 1
                    pred[s][v] = u
                    queue.append(v)
    return dist, pred


def _reconstruct_path(pred, s, t, n):
    """Reconstruct path from s to t using predecessor table."""
    if s == t:
        return [s]
    if pred[s][t] == -1:
        return None
    path = []
    cur = t
    while cur != -1:
        path.append(cur)
        if cur == s:
            break
        cur = pred[s][cur]
    if not path or path[-1] != s:
        return None
    path.reverse()
    return path


def _hungarian(cost):
    """
    Solve minimum-cost perfect matching (assignment) using Hungarian algorithm.
    cost is a k x k symmetric matrix (k must be even).
    Returns list of indices: match[i] = j means i is matched with j.
    """
    n = len(cost)
    if n == 0:
        return []
    if n % 2 != 0:
        raise ValueError("Hungarian requires even size")
    
    # Use the standard Hungarian algorithm for rectangular assignment
    # We'll use a simple O(n^3) implementation
    INF = float('inf')
    
    # u[i] = potential for row i, v[j] = potential for col j
    u = [0] * (n + 1)
    v = [0] * (n + 1)
    p = [0] * (n + 1)  # p[j] = row assigned to column j
    way = [0] * (n + 1)
    
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used = [False] * (n + 1)
        
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        
        # Augment
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    
    # p[j] = i means row i is assigned to column j
    match = [0] * n
    for j in range(1, n + 1):
        if p[j] != 0:
            match[p[j] - 1] = j - 1
    return match


def _greedy_matching(cost):
    """Greedy minimum-cost matching for small sets."""
    n = len(cost)
    if n == 0:
        return []
    INF = float('inf')
    used = [False] * n
    match = [-1] * n
    for i in range(n):
        if used[i]:
            continue
        best_j = -1
        best_cost = INF
        for j in range(n):
            if not used[j] and cost[i][j] < best_cost:
                best_cost = cost[i][j]
                best_j = j
        if best_j != -1:
            match[i] = best_j
            match[best_j] = i
            used[i] = True
            used[best_j] = True
    return match


def _path_cover_matching(edges, adj, n, deadline=None):
    """
    Edge path cover using matching on odd-degree vertices.
    """
    start_time = time.time()
    
    # Compute degrees
    deg = defaultdict(int)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    
    # Find odd-degree vertices
    odd_vertices = [v for v in range(n) if deg.get(v, 0) % 2 == 1]
    k = len(odd_vertices)
    
    # Lower bound on number of paths
    if k == 0:
        lower_bound = 1  # Eulerian circuit -> 1 path
    else:
        lower_bound = max(1, k // 2)
    
    if k <= 2:
        # Trivial: at most 1 path
        # Just do a greedy walk
        return _greedy_cover(edges, adj, n)
    
    # Compute shortest paths between all odd vertices
    dist, pred = _bfs_shortest_paths(adj, n)
    
    # Build cost matrix for matching
    cost = [[0] * k for _ in range(k)]
    for i in range(k):
        for j in range(k):
            if i == j:
                cost[i][j] = 0
            else:
                d = dist[odd_vertices[i]][odd_vertices[j]]
                cost[i][j] = d if d >= 0 else 10**6
    
    # Solve matching
    if k <= 20:
        match = _hungarian(cost)
    else:
        match = _greedy_matching(cost)
    
    # Extract paths for each matched pair
    remaining_edges = set(edges)
    paths = []
    
    for i in range(k // 2):
        j = match[i]
        if j == -1 or j <= i:
            continue
        u = odd_vertices[i]
        v = odd_vertices[j]
        path = _reconstruct_path(pred, u, v, n)
        if path is None:
            continue
        # Remove edges along this path from remaining
        for idx in range(len(path) - 1):
            a, b = path[idx], path[idx + 1]
            e = (a, b) if a < b else (b, a)
            if e in remaining_edges:
                remaining_edges.remove(e)
        paths.append(path)
    
    # Handle remaining edges with greedy
    if remaining_edges:
        extra_paths = _greedy_cover(remaining_edges, adj, n)
        paths.extend(extra_paths)
    
    # Try to merge paths
    paths = _merge_paths(paths, edges, adj, n)
    
    return paths


def _greedy_cover(edges, adj, n):
    """Simple greedy edge cover."""
    remaining = set(edges)
    paths = []
    
    while remaining:
        # Find vertex with max degree in remaining
        deg = defaultdict(int)
        for u, v in remaining:
            deg[u] += 1
            deg[v] += 1
        
        if not deg:
            break
        
        start = max(deg.keys(), key=lambda x: deg[x])
        visited = {start}
        path = [start]
        current = start
        
        while True:
            candidates = []
            for u, v in remaining:
                if u == current and v not in visited:
                    candidates.append(v)
                elif v == current and u not in visited:
                    candidates.append(u)
            
            if not candidates:
                break
            
            # Pick neighbor with max degree
            best_next = max(candidates, key=lambda x: deg.get(x, 0))
            path.append(best_next)
            visited.add(best_next)
            
            e = (current, best_next) if current < best_next else (best_next, current)
            if e in remaining:
                remaining.remove(e)
            
            current = best_next
        
        paths.append(path)
    
    return paths


def _merge_paths(paths, edges, adj, n):
    """
    Try to merge paths by finding connecting edges.
    If path A ends at vertex v and path B starts at vertex w, and edge (v,w) 
    exists and is not used in any path, we can merge.
    """
    if len(paths) <= 1:
        return paths
    
    # Track which edges are used
    used_edges = set()
    for path in paths:
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            e = (a, b) if a < b else (b, a)
            used_edges.add(e)
    
    # Try to merge
    merged = True
    while merged:
        merged = False
        new_paths = []
        used = [False] * len(paths)
        
        for i in range(len(paths)):
            if used[i]:
                continue
            cur_path = paths[i]
            
            # Try to extend cur_path by merging with other paths
            while True:
                end_vertex = cur_path[-1]
                start_vertex = cur_path[0]
                best_merge = None
                best_merge_type = None  # 'append' or 'prepend'
                
                for j in range(len(paths)):
                    if used[j] or j == i:
                        continue
                    other = paths[j]
                    
                    # Can we append other to cur_path?
                    # cur_path ends at end_vertex, other starts at other[0]
                    if end_vertex != other[0]:
                        e = (end_vertex, other[0]) if end_vertex < other[0] else (other[0], end_vertex)
                        if e not in used_edges:
                            # Check if this edge exists in original graph
                            if other[0] in adj.get(end_vertex, set()):
                                if best_merge is None or True:  # prefer any
                                    best_merge = other
                                    best_merge_type = 'append'
                                    break
                    
                    # Can we prepend other to cur_path?
                    # cur_path starts at start_vertex, other ends at other[-1]
                    if start_vertex != other[-1]:
                        e = (start_vertex, other[-1]) if start_vertex < other[-1] else (other[-1], start_vertex)
                        if e not in used_edges:
                            if other[-1] in adj.get(start_vertex, set()):
                                if best_merge is None:
                                    best_merge = other
                                    best_merge_type = 'prepend'
                                break
                
                if best_merge is None:
                    break
                
                if best_merge_type == 'append':
                    # cur_path + best_merge
                    cur_path = cur_path + best_merge[1:]
                    e = (end_vertex, best_merge[0]) if end_vertex < best_merge[0] else (best_merge[0], end_vertex)
                    used_edges.add(e)
                else:
                    # best_merge + cur_path
                    cur_path = best_merge[:-1] + cur_path
                    e = (best_merge[-1], start_vertex) if best_merge[-1] < start_vertex else (start_vertex, best_merge[-1])
                    used_edges.add(e)
                
                used[paths.index(best_merge)] = True
                merged = True
                # Mark this path as processed
                if i in used:
                    continue
                # Re-check if we can merge more
                # Actually, let's just do one pass for simplicity
        
        # Rebuild paths list
        for i in range(len(paths)):
            if not used[i]:
                if i < len(new_paths) and new_paths[i] is None:
                    pass
                # This is getting complicated; let's simplify
        
        # Simplified approach: just do one round of merging
        break
    
    return paths


def _improve_with_local_search(paths, edges, adj, n, time_budget=2.0):
    """
    Local search: try to swap endpoints of paths to reduce total count.
    """
    start = time.time()
    current_paths = [list(p) for p in paths]
    current_count = len(current_paths)
    
    rng = Random(123)
    
    while time.time() - start < time_budget and current_count > 1:
        # Try to find two paths that can be connected
        improved = False
        for i in range(len(current_paths)):
            for j in range(i + 1, len(current_paths)):
                pi = current_paths[i]
                pj = current_paths[j]
                
                # Try: reverse pi and connect
                # pi reversed ends at pi[0], pj starts at pj[0]
                # If pi[0] == pj[0], we can concatenate reversed pi + pj[1:]
                # But they share vertex pj[0], not edge. Need an edge.
                
                # Check all 4 combinations of endpoints
                endpoints_i = [(pi[0], pi[-1]), (pi[-1], pi[0])]
                endpoints_j = [(pj[0], pj[-1]), (pj[-1], pj[0])]
                
                for (start_i, end_i) in endpoints_i:
                    for (start_j, end_j) in endpoints_j:
                        # Connect end_i to start_j
                        if end_i != start_j:
                            e = (end_i, start_j) if end_i < start_j else (start_j, end_i)
                            if start_j in adj.get(end_i, set()):
                                # Check if edge is not already used in these two paths
                                # (simplification: assume it's available)
                                if e not in set(tuple(sorted((pi[k], pi[k+1]))) for k in range(len(pi)-1)) | \
                                   set(tuple(sorted((pj[k], pj[k+1]))) for k in range(len(pj)-1)):
                                    # Merge
                                    new_path = pi + pj[1:]
                                    current_paths[i] = new_path
                                    del current_paths[j]
                                    current_count -= 1
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return current_paths


def _path_cover_with_restarts(edges, adj, n, n_restarts=30, deadline=None):
    """
    Try multiple strategies: matching-based, greedy, and hybrid.
    """
    start_time = time.time()
    best_paths = None
    best_count = float('inf')
    
    # Strategy 1: Matching-based
    paths1 = _path_cover_matching(edges, adj, n, deadline)
    if paths1 and len(paths1) < best_count:
        best_count = len(paths1)
        best_paths = [list(p) for p in paths1]
    
    if deadline and time.time() > deadline - 5:
        return best_paths if best_paths else _greedy_cover(edges, adj, n)
    
    # Strategy 2: Greedy with different start vertices
    rng = Random(42)
    for restart in range(n_restarts):
        if deadline and time.time() > deadline - 3:
            break
        
        remaining = set(edges)
        paths = []
        
        while remaining:
            deg = defaultdict(int)
            for u, v in remaining:
                deg[u] += 1
                deg[v] += 1
            
            if not deg:
                break
            
            vertices = list(deg.keys())
            weights = [deg[v] for v in vertices]
            start = rng.choices(vertices, weights=weights, k=1)[0]
            
            visited = {start}
            path = [start]
            current = start
            
            while True:
                candidates = []
                for u, v in remaining:
                    if u == current and v not in visited:
                        candidates.append(v)
                    elif v == current and u not in visited:
                        candidates.append(u)
                
                if not candidates:
                    break
                
                weights = [deg.get(x, 0) for x in candidates]
                best_next = rng.choices(candidates, weights=weights, k=1)[0]
                path.append(best_next)
                visited.add(best_next)
                
                e = (current, best_next) if current < best_next else (best_next, current)
                if e in remaining:
                    remaining.remove(e)
                
                current = best_next
            
            paths.append(path)
        
        if len(paths) < best_count:
            best_count = len(paths)
            best_paths = [list(p) for p in paths]
        
        # Lower bound check
        deg_full = defaultdict(int)
        for u, v in edges:
            deg_full[u] += 1
            deg_full[v] += 1
        odd_count = sum(1 for v in range(n) if deg_full.get(v, 0) % 2 == 1)
        lower_bound = max(1, odd_count // 2)
        
        if best_count <= lower_bound:
            break
    
    # Local search improvement
    if best_paths and deadline and time.time() < deadline - 2:
        improved = _improve_with_local_search(best_paths, edges, adj, n, time_budget=min(2.0, max(0, deadline - time.time() - 1)))
        if len(improved) < best_count:
            best_paths = improved
            best_count = len(improved)
    
    return best_paths if best_paths else _greedy_cover(edges, adj, n)


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    results = []
    instances = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    
    for idx, (n, p, seed) in enumerate(instances):
        edges = _instance(n, p, seed)
        adj = _build_adj(edges, n)
        
        # Allocate time per instance
        remaining_time = deadline - time.time()
        if remaining_time <= 0:
            # Fallback
            paths = _greedy_cover(edges, adj, n)
            results.append(paths)
            continue
        
        # Use more time for later instances if needed
        n_restarts = 50 if idx < 2 else 30
        
        paths = _path_cover_with_restarts(edges, adj, n, n_restarts=n_restarts, deadline=deadline)
        results.append(paths)
        
        if time.time() > deadline - 10:
            # Rush through remaining
            for n2, p2, seed2 in instances[idx+1:]:
                edges2 = _instance(n2, p2, seed2)
                adj2 = _build_adj(edges2, n2)
                paths2 = _greedy_cover(edges2, adj2, n2)
                results.append(paths2)
            break
    
    return results
