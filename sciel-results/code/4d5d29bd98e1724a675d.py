"""Rewritten program: longest simple path via matching-based seed + local search."""
import random
from collections import deque

# Pinned graph data
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def _hopcroft_karp(adj, n):
    """Maximum cardinality bipartite matching via Hopcroft-Karp.
    adj: list of lists, adj[u] = list of v's (bipartite left->right edges)
    Returns: match array where match[u] = v or -1
    """
    INF = float('inf')
    match_left = [-1] * n
    match_right = [-1] * n
    dist = [0] * n
    
    def bfs():
        q = deque()
        found = False
        for u in range(n):
            if match_left[u] == -1:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = INF
        while q:
            u = q.popleft()
            for v in adj[u]:
                mu = match_right[v]
                if mu == -1:
                    found = True
                elif dist[mu] == INF:
                    dist[mu] = dist[u] + 1
                    q.append(mu)
        return found
    
    def dfs(u):
        for v in adj[u]:
            mu = match_right[v]
            if mu == -1 or (dist[mu] == dist[u] + 1 and dfs(mu)):
                match_left[u] = v
                match_right[v] = u
                return True
        dist[u] = INF
        return False
    
    matching_size = 0
    while bfs():
        for u in range(n):
            if match_left[u] == -1:
                if dfs(u):
                    matching_size += 1
    return match_left

def _build_paths_from_matching(adj, n, match_edges):
    """Given a list of matched edges (u,v), build disjoint paths by chaining.
    Returns list of paths (each path is a list of vertices).
    """
    # Build adjacency for the matching subgraph
    match_adj = [set() for _ in range(n)]
    matched_vertices = set()
    for u, v in match_edges:
        match_adj[u].add(v)
        match_adj[v].add(u)
        matched_vertices.add(u)
        matched_vertices.add(v)
    
    # Find connected components in the matching subgraph
    visited = set()
    paths = []
    for start in sorted(matched_vertices):
        if start in visited:
            continue
        # BFS to find the component
        component = []
        queue = deque([start])
        visited.add(start)
        while queue:
            node = queue.popleft()
            component.append(node)
            for nb in match_adj[node]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        
        # Find endpoints (vertices with degree 1 in matching)
        endpoints = [v for v in component if len(match_adj[v]) == 1]
        if not endpoints:
            # Isolated vertex or cycle
            if len(component) == 1:
                paths.append([component[0]])
            # For cycles, just take any ordering (break the cycle)
            else:
                # Find a path through the component
                paths.append([component[0]])
                for v in component[1:]:
                    if v in match_adj[paths[-1]]:
                        paths.append(v)
            continue
        
        # Build path from one endpoint to the other
        if len(endpoints) >= 2:
            path = [endpoints[0]]
            seen = {endpoints[0]}
            while len(path) < len(component):
                cur = path[-1]
                for nb in match_adj[cur]:
                    if nb not in seen:
                        path.append(nb)
                        seen.add(nb)
                        break
                else:
                    break
            paths.append(path)
        else:
            paths.append([endpoints[0]])
    
    return paths

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Step 1: Find maximum matching in the graph
    # Convert to bipartite: left = 0..n-1, right = 0..n-1
    # Edge (u,v) in original -> edge (u, v) in bipartite
    bipartite_adj = [list() for _ in range(n)]
    for u in range(n):
        for v in adj[u]:
            if u < v:  # Avoid duplicate edges
                bipartite_adj[u].append(v)
    
    match_left = _hopcroft_karp(bipartite_adj, n)
    
    # Extract matched edges
    match_edges = []
    for u in range(n):
        v = match_left[u]
        if v != -1:
            match_edges.append((u, v))
    
    # Step 2: Build paths from matching
    paths = _build_paths_from_matching(adj, n, match_edges)
    
    # Step 3: Try to merge paths by finding connecting edges
    def try_merge_paths(paths, adj, n):
        """Try to merge two paths by finding an edge between endpoints."""
        paths = [list(p) for p in paths]
        changed = True
        while changed:
            changed = False
            for i in range(len(paths)):
                for j in range(i + 1, len(paths)):
                    if not paths[i] or not paths[j]:
                        continue
                    pi = paths[i]
                    pj = paths[j]
                    # Try all 4 combinations of endpoints
                    combos = [
                        (pi[0], pj[0]),
                        (pi[0], pj[-1]),
                        (pi[-1], pj[0]),
                        (pi[-1], pj[-1]),
                    ]
                    merged = False
                    for a, b in combos:
                        if b in adj[a]:
                            # Merge: reverse pj if needed
                            if a == pi[0] and b == pj[-1]:
                                new_path = pj[::-1] + pi
                            elif a == pi[0] and b == pj[0]:
                                new_path = pi + pj[::-1]
                            elif a == pi[-1] and b == pj[0]:
                                new_path = pi + pj
                            else:  # a == pi[-1] and b == pj[-1]
                                new_path = pj + pi[::-1]
                            paths[i] = new_path
                            paths[j] = []
                            changed = True
                            merged = True
                            break
                    if merged:
                        break
                if changed:
                    break
            # Remove empty paths
            paths = [p for p in paths if p]
        return paths
    
    paths = try_merge_paths(paths, adj, n)
    
    # Step 4: Greedy extension of each path
    def extend_path(path, adj, n):
        path = list(path)
        path_set = set(path)
        changed = True
        while changed:
            changed = False
            # Extend at front
            if path:
                front = path[0]
                cands = [v for v in adj[front] if v not in path_set]
                if cands:
                    # Pick best candidate
                    best_v = min(cands, key=lambda x: (len(adj[x] - path_set), x))
                    path.insert(0, best_v)
                    path_set.add(best_v)
                    changed = True
            # Extend at back
            if path:
                back = path[-1]
                cands = [v for v in adj[back] if v not in path_set]
                if cands:
                    best_v = min(cands, key=lambda x: (len(adj[x] - path_set), x))
                    path.append(best_v)
                    path_set.add(best_v)
                    changed = True
        return path
    
    paths = [extend_path(p, adj, n) for p in paths]
    
    # Step 5: Try to merge extended paths again
    paths = try_merge_paths(paths, adj, n)
    paths = [extend_path(p, adj, n) for p in paths]
    paths = try_merge_paths(paths, adj, n)
    
    # Step 6: Local search - try to improve the longest path
    best_path = max(paths, key=len) if paths else []
    
    def local_search(path, adj, n, max_iter=500):
        """Improve path by trying to insert unvisited vertices."""
        path = list(path)
        path_set = set(path)
        unvisited = set(range(n)) - path_set
        
        for _ in range(max_iter):
            improved = False
            # Try inserting each unvisited vertex at each position
            for pos in range(len(path) + 1):
                if pos == 0:
                    # Insert at front
                    if not path:
                        continue
                    prev_v = path[0]
                    for u in unvisited:
                        if u in adj[prev_v]:
                            new_path = [u] + path
                            # Check if we can extend
                            new_set = set(new_path)
                            # Try to extend at back
                            back = new_path[-1]
                            cands = [v for v in adj[back] if v not in new_set]
                            if cands:
                                best_v = min(cands, key=lambda x: (len(adj[x] - new_set), x))
                                new_path.append(best_v)
                            if len(new_path) > len(path):
                                path = new_path
                                path_set = set(path)
                                unvisited = set(range(n)) - path_set
                                improved = True
                                break
                elif pos == len(path):
                    # Insert at back
                    prev_v = path[-1]
                    for u in unvisited:
                        if u in adj[prev_v]:
                            new_path = path + [u]
                            new_set = set(new_path)
                            # Try to extend at front
                            front = new_path[0]
                            cands = [v for v in adj[front] if v not in new_set]
                            if cands:
                                best_v = min(cands, key=lambda x: (len(adj[x] - new_set), x))
                                new_path.insert(0, best_v)
                            if len(new_path) > len(path):
                                path = new_path
                                path_set = set(path)
                                unvisited = set(range(n)) - path_set
                                improved = True
                                break
                else:
                    # Insert in middle: between path[pos-1] and path[pos]
                    a = path[pos - 1]
                    b = path[pos]
                    for u in unvisited:
                        if u in adj[a] and u in adj[b]:
                            new_path = path[:pos] + [u] + path[pos:]
                            new_set = set(new_path)
                            # Try to extend at both ends
                            front = new_path[0]
                            cands_f = [v for v in adj[front] if v not in new_set]
                            if cands_f:
                                best_v = min(cands_f, key=lambda x: (len(adj[x] - new_set), x))
                                new_path.insert(0, best_v)
                                new_set.add(best_v)
                            back = new_path[-1]
                            cands_b = [v for v in adj[back] if v not in new_set]
                            if cands_b:
                                best_v = min(cands_b, key=lambda x: (len(adj[x] - new_set), x))
                                new_path.append(best_v)
                            if len(new_path) > len(path):
                                path = new_path
                                path_set = set(path)
                                unvisited = set(range(n)) - path_set
                                improved = True
                                break
                if improved:
                    break
            if not improved:
                break
        return path
    
    best_path = local_search(best_path, adj, n)
    
    # Step 7: Final extension
    best_path = extend_path(best_path, adj, n)
    
    # Step 8: Try multiple random restarts with greedy + local search
    rng = random.Random(748)
    for restart in range(20):
        start = rng.randint(0, n - 1)
        path = [start]
        seen = {start}
        while True:
            cur = path[-1]
            cands = [v for v in adj[cur] if v not in seen]
            if not cands:
                break
            v = min(cands, key=lambda x: (len(adj[x] - seen), x))
            path.append(v)
            seen.add(v)
        path = extend_path(path, adj, n)
        path = local_search(path, adj, n, max_iter=200)
        if len(path) > len(best_path):
            best_path = path
    
    # Ensure validity
    final_path = [best_path[0]]
    for i in range(1, len(best_path)):
        if best_path[i] in adj[final_path[-1]]:
            final_path.append(best_path[i])
    
    if not final_path:
        final_path = [0]
    
    return final_path
