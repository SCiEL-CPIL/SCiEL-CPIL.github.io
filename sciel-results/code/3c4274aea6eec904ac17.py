import time
import sys

# Increase recursion limit
sys.setrecursionlimit(10000)

# P is the pinned graph data
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    deadline = time.time() + 85  # Safety margin
    
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
    
    # The longest path must be in the largest component
    # Sort components by size descending
    components.sort(key=len, reverse=True)
    
    best_path = [components[0][0]] if components else [0]
    best_len = len(best_path)
    
    # Precompute degree for heuristic
    degree = [len(adj[i]) for i in range(n)]
    
    def count_reachable(current, visited_set):
        """Count unvisited nodes reachable from current via unvisited nodes only."""
        bstack = [current]
        bseen = {current}
        count = 0
        while bstack:
            u = bstack.pop()
            for v in adj[u]:
                if v in visited_set:
                    continue
                if v not in bseen:
                    bseen.add(v)
                    count += 1
                    bstack.append(v)
        return count
    
    def dfs(path, current, visited):
        nonlocal best_path, best_len
        
        if len(path) > best_len:
            best_path = path[:]
            best_len = len(best_path)
        
        # Quick upper bound: total unvisited
        total_unvisited = n - len(visited)
        if len(path) + total_unvisited <= best_len:
            return
        
        # Get unvisited neighbors
        neighbors = [v for v in adj[current] if v not in visited]
        
        if not neighbors:
            return
        
        # Only compute reachable if it might help prune
        # If we have few neighbors, the reachable count is bounded by sum of their components
        # For efficiency, compute reachable only when we're close to best or have many options
        if len(neighbors) > 1 or len(path) > best_len - 10:
            reachable_count = count_reachable(current, visited)
            if len(path) + reachable_count <= best_len:
                return
        
        # Heuristic: Sort by number of unvisited neighbors (Warnsdorff-like)
        # Prefer nodes with fewer unvisited neighbors to avoid trapping
        def heuristic(v):
            unvis_nei = 0
            for w in adj[v]:
                if w not in visited:
                    unvis_nei += 1
            return unvis_nei
        
        neighbors.sort(key=heuristic)
        
        for v in neighbors:
            if time.time() >= deadline:
                return
            path.append(v)
            visited.add(v)
            dfs(path, v, visited)
            path.pop()
            visited.discard(v)
            if time.time() >= deadline:
                return
    
    # Start DFS from each node in the largest component
    largest_comp = components[0]
    
    # Optimization: If the component is a tree, we can find the diameter in O(N).
    # Let's check if it's a tree.
    # Edges in comp:
    comp_set = set(largest_comp)
    e_count = 0
    for u in largest_comp:
        for v in adj[u]:
            if v in comp_set and u < v:
                e_count += 1
                
    if e_count == len(largest_comp) - 1:
        # It's a tree! Find diameter.
        # Step 1: Pick any node, find farthest node A.
        # Step 2: Pick A, find farthest node B and path.
        def bfs_farthest(start, comp_set):
            dist = {start: 0}
            parent = {start: None}
            queue = [start]
            farthest = start
            max_dist = 0
            while queue:
                u = queue.pop(0)
                if dist[u] > max_dist:
                    max_dist = dist[u]
                    farthest = u
                for v in adj[u]:
                    if v in comp_set and v not in dist:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
            # Reconstruct path
            path = [farthest]
            while parent[farthest] is not None:
                farthest = parent[farthest]
                path.append(farthest)
            path.reverse()
            return path
        
        # Step 1
        start_node = largest_comp[0]
        path1 = bfs_farthest(start_node, comp_set)
        end_node_a = path1[-1]
        
        # Step 2
        path2 = bfs_farthest(end_node_a, comp_set)
        
        if len(path2) > best_len:
            best_path = path2
            best_len = len(best_path)
    else:
        # Not a tree, use DFS with pruning
        # Try starting from different nodes, prioritizing those with lower degree
        # to find longer paths faster
        
        # Sort starting nodes by degree (lower degree first for leaves)
        sorted_starts = sorted(largest_comp, key=lambda x: degree[x])
        
        for start in sorted_starts:
            if time.time() >= deadline:
                break
            path = [start]
            visited = {start}
            dfs(path, start, visited)
            if time.time() >= deadline:
                break
    
    return best_path

# EVOLVE-BLOCK-END
