import time
import sys
from collections import deque

def construct():
    start_time = time.time()
    time_limit = 85.0

    P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]]}

    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    # Convert to list for faster iteration
    adj_list = [list(s) for s in adj]

    # Find connected components
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

    components.sort(key=len, reverse=True)

    best_path = []

    # Function to get longest path in a tree (diameter)
    def get_longest_path_in_tree(comp, adj, root):
        if len(comp) == 1:
            return [comp[0]]
        
        # BFS from root to find farthest node u
        dist = {root: 0}
        parent = {root: None}
        q = deque([root])
        farthest = root
        while q:
            u = q.popleft()
            if dist[u] > dist[farthest]:
                farthest = u
            for v in adj[u]:
                if v in comp and v not in dist:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
        
        # BFS from u to find farthest node v
        dist2 = {farthest: 0}
        parent2 = {farthest: None}
        q = deque([farthest])
        farthest2 = farthest
        while q:
            u = q.popleft()
            if dist2[u] > dist2[farthest2]:
                farthest2 = u
            for v in adj[u]:
                if v in comp and v not in dist2:
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

    # Solve unicyclic component
    def solve_unicyclic(comp, adj):
        comp_set = set(comp)
        # Find the cycle by pruning leaves
        deg = {u: 0 for u in comp}
        for u in comp:
            for v in adj[u]:
                if v in comp_set:
                    deg[u] += 1
        
        remaining = set(comp)
        deg_rem = {u: deg[u] for u in comp}
        queue = deque([u for u in comp if deg_rem[u] == 1])
        
        while queue:
            u = queue.popleft()
            remaining.remove(u)
            for v in adj[u]:
                if v in remaining:
                    deg_rem[v] -= 1
                    if deg_rem[v] == 1:
                        queue.append(v)
        
        cycle_nodes = list(remaining)
        
        if not cycle_nodes:
            return get_longest_path_in_tree(comp, adj, comp[0])
        
        # Order the cycle nodes
        start = cycle_nodes[0]
        cycle_path = [start]
        prev = None
        curr = start
        visited_cycle = {start}
        
        next_node = None
        for v in adj[start]:
            if v in remaining and v != prev:
                next_node = v
                break
        
        curr = next_node
        while curr != start:
            cycle_path.append(curr)
            visited_cycle.add(curr)
            prev = curr
            next_node = None
            for v in adj[curr]:
                if v in remaining and v != prev:
                    next_node = v
                    break
            curr = next_node
            if curr is None:
                break
        
        cycle_len = len(cycle_path)
        cycle_edges = set()
        for i in range(cycle_len):
            u = cycle_path[i]
            v = cycle_path[(i + 1) % cycle_len]
            cycle_edges.add((u, v))
            cycle_edges.add((v, u))
            
        heights = [0] * cycle_len
        height_paths = [None] * cycle_len
        
        for i, u in enumerate(cycle_path):
            visited = {u}
            stack = [(u, 0, [u])]
            max_dist = 0
            farthest = u
            parent = {u: None}
            
            while stack:
                node, dist, path = stack.pop()
                if dist > max_dist:
                    max_dist = dist
                    farthest = node
                for v in adj[node]:
                    if v in comp_set and v not in visited:
                        if (node, v) in cycle_edges:
                            continue
                        visited.add(v)
                        parent[v] = node
                        stack.append((v, dist + 1, path + [v]))
                        
            heights[i] = max_dist
            if farthest == u:
                height_paths[i] = [u]
            else:
                path = []
                cur = farthest
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                height_paths[i] = path
        
        best_len = 0
        best_path_global = []
        
        # Consider paths entirely within one attached tree (diameter)
        for i, u in enumerate(cycle_path):
            tree_nodes = set()
            stack = [u]
            tree_nodes.add(u)
            while stack:
                node = stack.pop()
                for v in adj[node]:
                    if v in comp_set and v not in tree_nodes:
                        if (node, v) in cycle_edges:
                            continue
                        tree_nodes.add(v)
                        stack.append(v)
                        
            if len(tree_nodes) > 1:
                start_node = next(iter(tree_nodes))
                dist = {start_node: 0}
                parent = {start_node: None}
                q = deque([start_node])
                farthest1 = start_node
                while q:
                    node = q.popleft()
                    if dist[node] > dist[farthest1]:
                        farthest1 = node
                    for v in adj[node]:
                        if v in tree_nodes and v not in dist:
                            dist[v] = dist[node] + 1
                            parent[v] = node
                            q.append(v)
                            
                dist2 = {farthest1: 0}
                parent2 = {farthest1: None}
                q = deque([farthest1])
                farthest2 = farthest1
                while q:
                    node = q.popleft()
                    if dist2[node] > dist2[farthest2]:
                        farthest2 = node
                    for v in adj[node]:
                        if v in tree_nodes and v not in dist2:
                            dist2[v] = dist2[node] + 1
                            parent2[v] = node
                            q.append(v)
                            
                path = []
                cur = farthest2
                while cur is not None:
                    path.append(cur)
                    cur = parent2[cur]
                path.reverse()
                
                if len(path) > best_len:
                    best_len = len(path)
                    best_path_global = path
        
        # Consider paths using the cycle
        for i in range(cycle_len):
            for j in range(cycle_len):
                if i == j:
                    continue
                d1 = abs(i - j)
                d2 = cycle_len - d1
                d = min(d1, d2)
                
                total_len = heights[i] + heights[j] + d
                
                if total_len > best_len:
                    best_len = total_len
                    p1 = height_paths[i]
                    p2 = height_paths[j]
                    
                    if d1 == d:
                        if i < j:
                            cycle_seg = cycle_path[i:j+1]
                        else:
                            cycle_seg = cycle_path[i:] + cycle_path[:j+1]
                    else:
                        if i > j:
                            cycle_seg = cycle_path[j:i+1][::-1]
                        else:
                            cycle_seg = cycle_path[j:] + cycle_path[:i+1][::-1]
                    
                    full_path = p1 + cycle_seg[1:] + p2[1:]
                    
                    if len(full_path) > len(best_path_global):
                        best_path_global = full_path
        
        return best_path_global

    # DFS with branch and bound for general graphs
    def dfs_longest_path(start, adj_list, n, time_limit, start_time):
        best = [start]
        visited = [False] * n
        visited[start] = True
        path = [start]
        
        # Count unvisited neighbors for pruning
        def count_unvisited_options(u, visited):
            count = 0
            for v in adj_list[u]:
                if not visited[v]:
                    count += 1
            return count
        
        # Use iterative DFS with explicit stack to avoid recursion limits
        # Stack stores (node, path_tuple, visited_tuple) - but this is memory heavy
        # Instead, use recursion with sys.setrecursionlimit and careful pruning
        
        # Actually, let's use a recursive DFS with branch and bound
        # Since n=100, recursion depth can be up to 100, which is safe
        
        def dfs(u, path, visited):
            nonlocal best
            if time.time() - start_time > time_limit:
                return
            
            if len(path) > len(best):
                best = list(path)
            
            # Pruning: if max possible extension cannot beat best, return
            # Max possible extension is n - len(path)
            # But we can do better: count reachable unvisited nodes
            # For simplicity, use: if len(path) + (n - len(path)) <= len(best): return
            # This is trivial, so we need better pruning
            
            # Heuristic: only extend if there's a chance to beat best
            # Try neighbors in order of degree (highest first) to find long paths early
            neighbors = [v for v in adj_list[u] if not visited[v]]
            neighbors.sort(key=lambda x: len(adj_list[x]), reverse=True)
            
            for v in neighbors:
                if time.time() - start_time > time_limit:
                    return
                visited[v] = True
                path.append(v)
                dfs(v, path, visited)
                path.pop()
                visited[v] = False
        
        dfs(start, path, visited)
        return best

    # Process each component
    for comp in components:
        if time.time() - start_time > time_limit:
            break
            
        if len(comp) <= len(best_path):
            continue
            
        comp_set = set(comp)
        edge_count = sum(len(adj[u]) for u in comp) // 2
        
        if edge_count == len(comp) - 1:
            # Tree
            path = get_longest_path_in_tree(comp, adj, comp[0])
            if len(path) > len(best_path):
                best_path = path
        elif edge_count == len(comp):
            # Unicyclic
            path = solve_unicyclic(comp, adj)
            if len(path) > len(best_path):
                best_path = path
        else:
            # Complex component (multi-cyclic)
            # Use DFS with branch and bound
            # Try all start vertices in the component
            for start in comp:
                if time.time() - start_time > time_limit:
                    break
                path = dfs_longest_path(start, adj_list, n, time_limit, start_time)
                if len(path) > len(best_path):
                    best_path = path

    return best_path
