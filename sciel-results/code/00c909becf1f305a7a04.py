"""Rewritten program for longest simple path in pinned sparse graph."""
import time
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def _build_adj():
    adj = [set() for _ in range(P['n'])]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_components(adj):
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
                for nb in adj[node]:
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)
            components.append(comp)
    return components

def _is_tree(comp, adj):
    """Check if the induced subgraph on comp is a tree."""
    n_comp = len(comp)
    if n_comp <= 1:
        return True
    edge_count = 0
    comp_set = set(comp)
    for v in comp:
        for nb in adj[v]:
            if nb in comp_set and v < nb:
                edge_count += 1
    return edge_count == n_comp - 1

def _tree_diameter(comp, adj):
    """Find diameter of a tree using two BFS passes."""
    if len(comp) <= 1:
        return comp[:]
    
    comp_set = set(comp)
    
    def bfs(start):
        dist = {start: 0}
        parent = {start: None}
        q = deque([start])
        farthest = start
        while q:
            node = q.popleft()
            for nb in adj[node]:
                if nb in comp_set and nb not in dist:
                    dist[nb] = dist[node] + 1
                    parent[nb] = node
                    if dist[nb] > dist[farthest]:
                        farthest = nb
                    q.append(nb)
        # Reconstruct path
        path = []
        cur = farthest
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        return path
    
    # First BFS to find one endpoint
    path1 = bfs(comp[0])
    endpoint1 = path1[-1]
    # Second BFS from endpoint1
    path2 = bfs(endpoint1)
    return path2

def _longest_path_in_component(comp, adj, deadline):
    """Find longest simple path in a connected component."""
    if time.time() > deadline:
        return []
    
    n_comp = len(comp)
    if n_comp == 0:
        return []
    if n_comp == 1:
        return comp[:]
    
    comp_set = set(comp)
    
    # Check if it's a tree - if so, use diameter algorithm
    if _is_tree(comp, adj):
        return _tree_diameter(comp, adj)
    
    # For small components with cycles, use DFS with backtracking
    if n_comp <= 20:
        best_path = []
        
        def dfs(node, path, visited):
            nonlocal best_path
            if time.time() > deadline:
                return
            if len(path) > len(best_path):
                best_path = path[:]
            for nb in sorted(adj[node]):
                if nb in comp_set and nb not in visited:
                    visited.add(nb)
                    path.append(nb)
                    dfs(nb, path, visited)
                    path.pop()
                    visited.discard(nb)
        
        for start in comp:
            if time.time() > deadline:
                break
            dfs(start, [start], {start})
        
        return best_path
    
    # For larger components, use IDDFS with Warnsdorff heuristic
    # First, try greedy with Warnsdorff from each starting vertex
    best_greedy = []
    
    for start in comp:
        if time.time() > deadline:
            break
        path = [start]
        visited = {start}
        while True:
            cur = path[-1]
            candidates = [nb for nb in adj[cur] if nb in comp_set and nb not in visited]
            if not candidates:
                break
            # Warnsdorff: pick the neighbor with fewest unvisited neighbors
            def warnsdorff_key(nb):
                unvisited_count = sum(1 for x in adj[nb] if x in comp_set and x not in visited)
                return (unvisited_count, nb)
            next_v = min(candidates, key=warnsdorff_key)
            path.append(next_v)
            visited.add(next_v)
        
        if len(path) > len(best_greedy):
            best_greedy = path[:]
    
    if len(best_greedy) >= n_comp - 1:
        return best_greedy
    
    # Try to extend and improve using local search
    # Try 2-opt style improvements: try swapping segments
    best_path = best_greedy[:]
    
    # Try starting from different vertices with slight variations
    for start in comp:
        if time.time() > deadline:
            break
        # Try with different tie-breaking in Warnsdorff
        for offset in range(3):
            path = [start]
            visited = {start}
            while True:
                cur = path[-1]
                candidates = [nb for nb in adj[cur] if nb in comp_set and nb not in visited]
                if not candidates:
                    break
                def warnsdorff_key2(nb):
                    unvisited_count = sum(1 for x in adj[nb] if x in comp_set and x not in visited)
                    return (unvisited_count, (nb + offset) % 100)
                next_v = min(candidates, key=warnsdorff_key2)
                path.append(next_v)
                visited.add(next_v)
            
            if len(path) > len(best_path):
                best_path = path[:]
    
    # Try DFS with depth limit for improvement
    if n_comp <= 35:
        def dfs_improve(node, path, visited, max_depth):
            nonlocal best_path
            if time.time() > deadline:
                return
            if len(path) > len(best_path):
                best_path = path[:]
            if len(path) >= max_depth:
                return
            for nb in sorted(adj[node]):
                if nb in comp_set and nb not in visited:
                    visited.add(nb)
                    path.append(nb)
                    dfs_improve(nb, path, visited, max_depth)
                    path.pop()
                    visited.discard(nb)
        
        for start in comp:
            if time.time() > deadline:
                break
            dfs_improve(start, [start], {start}, n_comp)
    
    return best_path

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    adj = _build_adj()
    components = _find_components(adj)
    
    best_path = []
    
    for comp in components:
        if time.time() > deadline:
            break
        path = _longest_path_in_component(comp, adj, deadline)
        if len(path) > len(best_path):
            best_path = path
    
    # Fallback: if no path found, return single vertex
    if not best_path:
        best_path = [0]
    
    return best_path
# EVOLVE-BLOCK-END
