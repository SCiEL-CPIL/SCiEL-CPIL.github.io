import time
from collections import deque

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
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
                for nei in adj_list[node]:
                    if not visited[nei]:
                        visited[nei] = True
                        stack.append(nei)
            components.append(comp)
    
    largest_comp = max(components, key=len)
    comp_size = len(largest_comp)
    comp_set = set(largest_comp)
    
    if comp_size == 0:
        return [0]
    if comp_size == 1:
        return largest_comp[:]
    
    # Compute degrees within the component
    deg = {u: len([v for v in adj_list[u] if v in comp_set]) for u in largest_comp}
    e_count = sum(deg[u] for u in largest_comp) // 2
    is_tree = (e_count == comp_size - 1)
    
    best_path = []
    best_len = 0
    
    deadline = time.time() + 85.0
    
    if is_tree:
        # Tree: longest path is the diameter
        def bfs_farthest(start):
            dist = {start: 0}
            parent = {start: None}
            q = deque([start])
            farthest = start
            max_d = 0
            while q:
                u = q.popleft()
                if dist[u] > max_d:
                    max_d = dist[u]
                    farthest = u
                for v in adj_list[u]:
                    if v in comp_set and v not in dist:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
            return farthest, parent
        
        u, _ = bfs_farthest(largest_comp[0])
        v, parent = bfs_farthest(u)
        
        path = []
        curr = v
        while curr is not None:
            path.append(curr)
            curr = parent[curr]
        path.reverse()
        return path
    
    # For non-tree: use DFS with connectivity pruning
    # Precompute adjacency within component as sets for fast lookup
    comp_adj = {u: set(v for v in adj_list[u] if v in comp_set) for u in largest_comp}
    
    # Warnsdorff ordering: prefer neighbors with fewer onward options
    def onward_count(v, visited_set, current):
        count = 0
        for w in comp_adj[v]:
            if w not in visited_set and w != current:
                count += 1
        return count
    
    # Connectivity check: can all unvisited vertices be reached from either endpoint?
    # This is expensive, so we use a lighter check: 
    # Check that unvisited vertices adjacent to path endpoints can reach all other unvisited
    def connectivity_feasible(path, visited_set):
        if len(path) == 0:
            return True
        unvisited = [v for v in largest_comp if v not in visited_set]
        if not unvisited:
            return True
        
        # The unvisited subgraph must be connected when we consider 
        # edges from path endpoints to unvisited
        # Light check: BFS from all unvisited vertices that are adjacent to path endpoints
        # Actually, simpler: check that the unvisited induced subgraph is connected
        # OR that it can be reached from at least one endpoint
        
        # Even simpler pruning: if removing visited vertices disconnects the unvisited
        # into multiple components, and none of those components touches both endpoints,
        # it's infeasible. But this is complex.
        
        # Lighter: just check that the unvisited subgraph is connected
        if len(unvisited) <= 1:
            return True
        
        # BFS within unvisited
        start = unvisited[0]
        seen = {start}
        q = deque([start])
        while q:
            u = q.popleft()
            for v in comp_adj[u]:
                if v not in visited_set and v not in seen:
                    seen.add(v)
                    q.append(v)
        
        if len(seen) != len(unvisited):
            return False
        return True
    
    def dfs_path(curr, path, visited_set, time_limit):
        nonlocal best_path, best_len
        
        if time.time() > time_limit:
            return False
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
        
        if len(path) == comp_size:
            return True
        
        # Pruning: if remaining vertices can't extend beyond best, stop
        remaining = comp_size - len(visited_set)
        if len(path) + remaining <= best_len:
            return False
        
        neighbors = [v for v in comp_adj[curr] if v not in visited_set]
        
        if not neighbors:
            return False
        
        # Sort by Warnsdorff: fewest onward moves first
        neighbors.sort(key=lambda x: (onward_count(x, visited_set, curr), deg[x]))
        
        for v in neighbors:
            path.append(v)
            visited_set.add(v)
            
            # Connectivity pruning: check if remaining is feasible
            if len(visited_set) < comp_size and not connectivity_feasible(path, visited_set):
                path.pop()
                visited_set.remove(v)
                continue
            
            if dfs_path(v, path, visited_set, time_limit):
                path.pop()
                visited_set.remove(v)
                return True
            
            path.pop()
            visited_set.remove(v)
            
            if time.time() > time_limit:
                return False
        
        return False
    
    # Try DFS from promising start vertices
    # Low-degree vertices are more likely to be path endpoints
    start_nodes = sorted(largest_comp, key=lambda x: (deg[x], x))
    
    # Also include some high-degree vertices
    high_deg = sorted(largest_comp, key=lambda x: (-deg[x], x))[:5]
    dfs_starts = list(set(start_nodes[:20] + high_deg))
    
    for start in dfs_starts:
        if time.time() > deadline - 3:
            break
        
        path = [start]
        visited_set = {start}
        
        dfs_path(start, path, visited_set, deadline - 3)
        
        if best_len == comp_size:
            break
    
    if not best_path:
        best_path = [largest_comp[0]]
    
    return best_path
