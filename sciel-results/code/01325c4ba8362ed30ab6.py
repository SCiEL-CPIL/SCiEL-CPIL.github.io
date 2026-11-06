import time
import sys
from collections import deque

# Increase recursion limit
sys.setrecursionlimit(10000)

# P is the pinned graph data
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    deadline = time.time() + 85  # Safety margin before 90s kill
    
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find connected components
    visited_comp = [False] * n
    components = []
    for i in range(n):
        if not visited_comp[i]:
            stack = [i]
            visited_comp[i] = True
            comp = []
            while stack:
                node = stack.pop()
                comp.append(node)
                for nei in adj[node]:
                    if not visited_comp[nei]:
                        visited_comp[nei] = True
                        stack.append(nei)
            components.append(comp)
    
    # Sort components by size descending
    components.sort(key=len, reverse=True)
    
    best_path = [components[0][0]] if components else [0]
    best_len = len(best_path)
    
    largest_comp = components[0]
    comp_set = set(largest_comp)
    comp_size = len(largest_comp)
    
    # Count edges in the largest component
    e_count = 0
    for u in largest_comp:
        for v in adj[u]:
            if v in comp_set and u < v:
                e_count += 1
    
    # If it's a tree, find diameter in O(N)
    if e_count == comp_size - 1:
        def bfs_farthest(start, comp_set):
            dist = {start: 0}
            parent = {start: None}
            queue = deque([start])
            farthest = start
            max_dist = 0
            while queue:
                u = queue.popleft()
                if dist[u] > max_dist:
                    max_dist = dist[u]
                    farthest = u
                for v in adj[u]:
                    if v in comp_set and v not in dist:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
            path = [farthest]
            while parent[farthest] is not None:
                farthest = parent[farthest]
                path.append(farthest)
            path.reverse()
            return path
        
        path1 = bfs_farthest(largest_comp[0], comp_set)
        end_node_a = path1[-1]
        path2 = bfs_farthest(end_node_a, comp_set)
        
        if len(path2) > best_len:
            best_path = path2
            best_len = len(path2)
        
        return best_path
    
    # Precompute degrees within the component
    deg = {u: sum(1 for v in adj[u] if v in comp_set) for u in largest_comp}
    
    # Matching-based slack computation
    # For each vertex, compute a "slack" value that indicates how constrained it is
    # Low slack means the vertex is a bottleneck and should be visited early or carefully
    
    def compute_slack(comp_set, adj):
        """
        Compute matching slack for each vertex.
        A vertex with low slack is more constrained (fewer alternative neighbors).
        We use a simple approximation: slack = deg - 2 for internal vertices,
        but for a path, the two endpoints have slack = deg - 1.
        Actually, let's use a bipartite matching approach to find which vertices
        are critical.
        
        For simplicity and speed, we'll use:
        slack[v] = number of unvisited neighbors (initially just degree)
        But we want to identify vertices that, if left unvisited until late,
        would trap the path.
        
        A better heuristic: vertices with degree 2 are the most constrained
        (they must be internal to the path, with both neighbors used).
        Vertices with degree 1 can only be endpoints.
        """
        slack = {}
        for v in comp_set:
            d = deg[v]
            if d == 1:
                slack[v] = 0  # Must be an endpoint
            elif d == 2:
                slack[v] = 1  # Very constrained, likely internal
            else:
                slack[v] = d - 2  # More flexibility
        return slack
    
    slack = compute_slack(comp_set, adj)
    
    # Greedy path with slack-aware heuristic
    def greedy_path_slack(start, visited):
        """Greedy extension from start, using slack to guide choices."""
        path = [start]
        current = start
        visited.add(start)
        while True:
            neighbors = [v for v in adj[current] if v not in visited and v in comp_set]
            if not neighbors:
                break
            # Pick neighbor with lowest slack (most constrained first)
            # This is a Warnsdorff-like heuristic adapted for slack
            neighbors.sort(key=lambda v: (slack[v], deg[v]))
            current = neighbors[0]
            visited.add(current)
            path.append(current)
        return path
    
    # Try greedy from each node to get initial best
    for start in largest_comp:
        if time.time() >= deadline:
            break
        visited = {start}
        path = greedy_path_slack(start, visited)
        if len(path) > best_len:
            best_path = path
            best_len = len(path)
    
    # DFS with pruning and slack-based ordering
    def reachable_from(current, visited):
        """Count unvisited nodes in comp_set reachable from current via unvisited nodes."""
        if current not in comp_set:
            return 0
        stack = [current]
        seen = {current}
        count = 0
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v in visited or v not in comp_set:
                    continue
                if v not in seen:
                    seen.add(v)
                    count += 1
                    stack.append(v)
        return count
    
    search_done = [False]
    
    def dfs_search(path, current, visited, depth):
        nonlocal best_path, best_len, search_done
        if search_done[0] or time.time() >= deadline:
            search_done[0] = True
            return
        
        if len(path) > best_len:
            best_path = path[:]
            best_len = len(path)
        
        # Pruning: can we possibly beat best_len?
        if best_len >= comp_size:
            search_done[0] = True
            return
        
        remaining = comp_size - len(visited)
        if len(path) + remaining <= best_len:
            return
        
        reach = reachable_from(current, visited)
        if len(path) + reach <= best_len:
            return
        
        # Get unvisited neighbors
        neighbors = [v for v in adj[current] if v not in visited and v in comp_set]
        if not neighbors:
            return
        
        # Sort by slack (lowest first) to prioritize constrained vertices
        # This helps avoid trapping low-degree vertices
        neighbors.sort(key=lambda v: (slack[v], deg[v]))
        
        for v in neighbors:
            if search_done[0] or time.time() >= deadline:
                search_done[0] = True
                return
            path.append(v)
            visited.add(v)
            dfs_search(path, v, visited, depth + 1)
            path.pop()
            visited.discard(v)
    
    # Start DFS from nodes sorted by slack (lowest slack first, as they are more constrained)
    starts = sorted(largest_comp, key=lambda x: (slack[x], -deg[x]))
    
    # Try DFS from top-k starts
    num_starts = min(len(starts), 20)
    
    for start in starts[:num_starts]:
        if time.time() >= deadline:
            break
        if search_done[0]:
            break
        path = [start]
        visited = {start}
        dfs_search(path, start, visited, 1)
    
    # If we have time, try more starts
    if not search_done[0] and time.time() < deadline - 10:
        for start in starts[num_starts:]:
            if time.time() >= deadline:
                break
            if search_done[0]:
                break
            path = [start]
            visited = {start}
            dfs_search(path, start, visited, 1)
    
    # Final check: if best_len == comp_size, we found a Hamiltonian path
    # Otherwise, return best_path
    
    return best_path

# EVOLVE-BLOCK-END
