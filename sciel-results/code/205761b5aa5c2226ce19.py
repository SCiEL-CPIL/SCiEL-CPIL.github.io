import time
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_components(adj, n):
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

def _bfs_diameter_path(adj, comp, comp_set):
    """Find the longest path in a tree component using BFS diameter."""
    if len(comp) == 1:
        return list(comp)
    
    # BFS from arbitrary node to find farthest
    start = comp[0]
    dist = {start: 0}
    parent = {start: None}
    q = deque([start])
    farthest = start
    while q:
        u = q.popleft()
        if dist[u] > dist[farthest]:
            farthest = u
        for v in adj[u]:
            if v in comp_set and v not in dist:
                dist[v] = dist[u] + 1
                parent[v] = u
                q.append(v)
    
    # BFS from farthest
    dist2 = {farthest: 0}
    parent2 = {farthest: None}
    q = deque([farthest])
    farthest2 = farthest
    while q:
        u = q.popleft()
        if dist2[u] > dist2[farthest2]:
            farthest2 = u
        for v in adj[u]:
            if v in comp_set and v not in dist2:
                dist2[v] = dist2[u] + 1
                parent2[v] = u
                q.append(v)
    
    # Reconstruct path
    path = []
    cur = farthest2
    while cur is not None:
        path.append(cur)
        cur = parent2[cur]
    path.reverse()
    return path

def _greedy_path(adj, start, n, max_steps=100000):
    """Greedy path: at each step go to neighbor with fewest unvisited neighbors."""
    path = [start]
    visited = {start}
    steps = 0
    while steps < max_steps:
        cur = path[-1]
        cands = [nb for nb in adj[cur] if nb not in visited]
        if not cands:
            break
        cands.sort(key=lambda x: (len(adj[x] - visited), x))
        nxt = cands[0]
        path.append(nxt)
        visited.add(nxt)
        steps += 1
    return path

def _try_extend_path(adj, path, n, deadline):
    """Try to extend the path from both ends. Returns improved path."""
    improved = True
    path_set = set(path)
    while improved:
        improved = False
        if time.time() > deadline:
            break
        # Try extending the end
        end = path[-1]
        cands = [nb for nb in adj[end] if nb not in path_set]
        if cands:
            cands.sort(key=lambda x: len(adj[x] & path_set))
            path.append(cands[0])
            path_set.add(cands[0])
            improved = True
            continue
        # Try extending the beginning
        start = path[0]
        cands = [nb for nb in adj[start] if nb not in path_set]
        if cands:
            cands.sort(key=lambda x: len(adj[x] & path_set))
            path.insert(0, cands[0])
            path_set.add(cands[0])
            improved = True
            continue
    return path

def _insert_vertex(adj, path, vertex, n):
    """Try to insert vertex into the path. Returns new path or None."""
    if vertex in set(path):
        return None
    # Check if vertex can be inserted at the start
    if vertex in adj[path[0]]:
        return [vertex] + path
    # Check if vertex can be inserted at the end
    if vertex in adj[path[-1]]:
        return path + [vertex]
    # Check internal insertion
    for i in range(len(path) - 1):
        a, b = path[i], path[i+1]
        if vertex in adj[a] and vertex in adj[b]:
            new_path = path[:i+1] + [vertex] + path[i+1:]
            return new_path
    return None

def _dfs_path(adj, comp, comp_set, n, deadline):
    """DFS to find a long path in the component."""
    best = []
    comp_list = list(comp)
    
    # Try multiple starting points
    starts = comp_list[:min(30, len(comp_list))]
    
    for start in starts:
        if time.time() > deadline:
            break
        path = _greedy_path(adj, start, n)
        if len(path) > len(best):
            best = path[:]
    
    return best

def _branch_and_bound_path(adj, comp, comp_set, n, deadline):
    """
    Branch and bound to find a long path in the component.
    Uses DFS with pruning: if the remaining unvisited vertices can't 
    beat the current best, prune.
    """
    if len(comp) <= 1:
        return list(comp)
    
    best_path = []
    best_len = 0
    visited = set()
    path = []
    
    # Use iterative deepening or just DFS with good heuristics
    # For small components, exact B&B works well
    
    def dfs(node, depth):
        nonlocal best_path, best_len, visited, path
        
        if time.time() > deadline:
            return
        
        path.append(node)
        visited.add(node)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
        
        # Try to extend
        cands = [nb for nb in adj[node] if nb in comp_set and nb not in visited]
        if not cands:
            path.pop()
            visited.remove(node)
            return
        
        # Sort candidates: prefer those with fewer unvisited neighbors (Warnsdorff-like)
        cands.sort(key=lambda x: len(adj[x] & comp_set - visited))
        
        for nb in cands:
            if time.time() > deadline:
                break
            dfs(nb, depth + 1)
        
        path.pop()
        visited.remove(node)
    
    # Try from each vertex in the component as start
    for start in comp:
        if time.time() > deadline:
            break
        path = []
        visited = set()
        dfs(start, 0)
    
    return best_path

def _try_swap(adj, path, n, deadline):
    """Try to improve path by removing a vertex and reinserting elsewhere, or by 2-opt style swaps."""
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        path_set = set(path)
        # Try to find a vertex not in path that can be inserted
        for v in range(n):
            if v not in path_set:
                new_path = _insert_vertex(adj, path, v, n)
                if new_path is not None:
                    path = new_path
                    improved = True
                    break
        if improved:
            continue
        # Try 2-opt: reverse a segment and see if we can then extend
        for i in range(len(path) - 1):
            if time.time() > deadline:
                return path
            # Skip if reversing would break edges
            if i > 0 and path[i-1] not in adj[path[i+1]]:
                continue
            if i + 1 < len(path) - 1 and path[i+2] not in adj[path[i]]:
                continue
            new_path = path[:i+1] + path[i+1:][::-1] + path[i+2:] if i+2 < len(path) else path[:i+1] + path[i+1:][::-1]
            if len(new_path) != len(path):
                continue
            # Verify it's still a valid path
            valid = True
            for j in range(len(new_path) - 1):
                if new_path[j+1] not in adj[new_path[j]]:
                    valid = False
                    break
            if not valid:
                continue
            # Try extending the new path
            new_path_set = set(new_path)
            end_cands = [nb for nb in adj[new_path[-1]] if nb not in new_path_set]
            start_cands = [nb for nb in adj[new_path[0]] if nb not in new_path_set]
            if end_cands or start_cands:
                # Check if this reversal enables new extensions
                # Try extending
                if end_cands:
                    end_cands.sort(key=lambda x: len(adj[x] & new_path_set))
                    new_path.append(end_cands[0])
                if start_cands:
                    start_cands.sort(key=lambda x: len(adj[x] & new_path_set))
                    new_path.insert(0, start_cands[0])
                if len(new_path) > len(path):
                    path = new_path
                    improved = True
                    break
    return path

def construct():
    n = P['n']
    adj = _build_adj()
    
    deadline = time.time() + 75.0
    
    # Find components
    components = _find_components(adj, n)
    components.sort(key=len, reverse=True)
    
    # Get initial path from largest component
    best_path = []
    for comp in components:
        if len(comp) <= len(best_path):
            continue
        # Check if it's a tree
        comp_set = set(comp)
        edge_count = sum(len(adj[u]) for u in comp) // 2
        if edge_count == len(comp) - 1:
            # Tree: use BFS diameter
            if len(comp) == 1:
                if len(comp) > len(best_path):
                    best_path = list(comp)
                continue
            path = _bfs_diameter_path(adj, comp, comp_set)
            if len(path) > len(best_path):
                best_path = path
            continue
        # Cyclic component: try B&B for small components, greedy for large
        if time.time() > deadline:
            break
        if len(comp) <= 25:
            # Small enough for exact B&B
            bb_path = _branch_and_bound_path(adj, comp, comp_set, n, deadline)
            if len(bb_path) > len(best_path):
                best_path = bb_path
        else:
            # Large: use greedy from multiple starts
            dfs_best = _dfs_path(adj, comp, comp_set, n, deadline)
            if len(dfs_best) > len(best_path):
                best_path = dfs_best
    
    # Now optimize: iteratively try to grow the path
    best_path = _try_extend_path(adj, best_path, n, deadline)
    
    # Iterative improvement loop
    iteration = 0
    while time.time() < deadline - 2.0:
        iteration += 1
        # Try inserting vertices
        path_set = set(best_path)
        inserted = False
        for v in range(n):
            if v not in path_set:
                new_path = _insert_vertex(adj, best_path, v, n)
                if new_path is not None:
                    best_path = new_path
                    path_set = set(best_path)
                    inserted = True
                    break
        if not inserted:
            # Try 2-opt style improvements
            new_best = _try_swap(adj, best_path, n, deadline)
            if len(new_best) > len(best_path):
                best_path = new_best
            else:
                # No improvement possible via current methods
                # Try a fresh greedy from a different start
                if iteration < 10:
                    fresh = _greedy_path(adj, best_path[0], n)
                    fresh = _try_extend_path(adj, fresh, n, deadline)
                    if len(fresh) > len(best_path):
                        best_path = fresh
                    else:
                        break
        else:
            # Re-extend after insertion
            best_path = _try_extend_path(adj, best_path, n, deadline)
    
    # Final verification
    if len(best_path) == 0:
        # Should not happen, but return a single vertex
        best_path = [0]
    
    # Verify validity
    for i in range(len(best_path) - 1):
        assert best_path[i+1] in adj[best_path[i]], f"Invalid edge at position {i}"
    assert len(set(best_path)) == len(best_path), "Path has repeated vertices"
    
    return best_path
# EVOLVE-BLOCK-END
