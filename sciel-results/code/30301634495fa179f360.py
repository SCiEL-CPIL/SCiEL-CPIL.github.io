import time
import sys
from collections import deque
import itertools

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
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
                u = stack.pop()
                comp.append(u)
                for v in adj[u]:
                    if not visited_comp[v]:
                        visited_comp[v] = True
                        stack.append(v)
            components.append(comp)
    
    best_path = []
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    def check_time():
        return time.time() > deadline
    
    # Function to find longest path in a tree component using BFS diameter
    def find_longest_path_in_tree(comp, local_adj):
        if len(comp) <= 1:
            return list(comp)
        
        def bfs_farthest(start):
            dist = {start: 0}
            parent = {start: -1}
            queue = deque([start])
            farthest = start
            max_dist = 0
            while queue:
                u = queue.popleft()
                if dist[u] > max_dist:
                    max_dist = dist[u]
                    farthest = u
                for v in local_adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
            return farthest, dist, parent
        
        start_node = comp[0]
        A, dist1, parent1 = bfs_farthest(start_node)
        B, dist2, parent2 = bfs_farthest(A)
        
        path = [B]
        cur = B
        while cur != A:
            cur = parent2[cur]
            path.append(cur)
        path.reverse()
        return path
    
    # Function to find longest path using DFS with pruning for general graphs
    def find_longest_path_dfs(comp, local_adj, time_budget=None):
        if time_budget is None:
            time_budget = deadline
        
        best_local = []
        n_comp = len(comp)
        
        # Map local indices
        comp_set = set(comp)
        
        def dfs(u, visited, path, start_idx):
            nonlocal best_local
            if check_time():
                return False  # Signal to stop
            if len(path) > len(best_local):
                best_local = path[:]
            
            neighbors = [v for v in local_adj[u] if v not in visited]
            if not neighbors:
                return True
            
            # Heuristic: try neighbors with fewer unvisited neighbors first (Warnsdorff)
            def get_priority(v):
                unvisited_count = sum(1 for w in local_adj[v] if w not in visited)
                return unvisited_count
            
            neighbors.sort(key=get_priority)
            
            for v in neighbors:
                visited.add(v)
                path.append(v)
                result = dfs(v, visited, path, start_idx)
                path.pop()
                visited.remove(v)
                if result:
                    return True
                if check_time():
                    return True
            
            return False
        
        # Try starting from each node, prioritizing low-degree nodes
        starts = sorted(comp, key=lambda x: len(local_adj[x]))
        
        for start in starts:
            if check_time():
                break
            visited = {start}
            path = [start]
            dfs(start, visited, path, start)
        
        return best_local
    
    # Function to find longest path using iterative deepening
    def find_longest_path_iterative(comp, local_adj, max_depth=None):
        if max_depth is None:
            max_depth = len(comp)
        
        best_local = []
        comp_set = set(comp)
        
        # Try increasing path lengths from max possible down
        for target_len in range(max_depth, 1, -1):
            if check_time():
                break
            
            found = False
            # Try all starting nodes
            starts = sorted(comp, key=lambda x: len(local_adj[x]))
            
            for start in starts:
                if found or check_time():
                    break
                
                def dfs_iter(u, visited, path, remaining):
                    nonlocal found
                    if check_time():
                        return True
                    if len(path) == target_len:
                        if len(path) > len(best_local):
                            best_local = path[:]
                        found = True
                        return True
                    
                    if remaining == 0:
                        return False
                    
                    neighbors = [v for v in local_adj[u] if v not in visited]
                    if not neighbors:
                        return False
                    
                    # Pruning: if we can't reach target_len, return early
                    if len(path) + remaining < target_len:
                        return False
                    
                    # Heuristic ordering
                    def get_priority(v):
                        unvisited_count = sum(1 for w in local_adj[v] if w not in visited)
                        return unvisited_count
                    
                    neighbors.sort(key=get_priority)
                    
                    for v in neighbors:
                        visited.add(v)
                        path.append(v)
                        result = dfs_iter(v, visited, path, remaining - 1)
                        path.pop()
                        visited.remove(v)
                        if result:
                            return True
                    
                    return False
                
                visited = {start}
                path = [start]
                dfs_iter(start, visited, path, target_len - 1)
            
            if found:
                break
        
        return best_local
    
    # Process each component
    for comp in sorted(components, key=len, reverse=True):
        if check_time():
            break
        
        # Build local adjacency
        local_adj = {u: set() for u in comp}
        for u in comp:
            for v in adj[u]:
                if v in local_adj:
                    local_adj[u].add(v)
        
        # Check if it's a tree
        edge_count = 0
        for u in comp:
            edge_count += len(local_adj[u])
        edge_count //= 2
        
        if edge_count == len(comp) - 1:
            # It's a tree - use BFS diameter
            path = find_longest_path_in_tree(comp, local_adj)
            if len(path) > len(best_path):
                best_path = path
        else:
            # Has cycles - use DFS with pruning
            # For small components, try iterative deepening first
            if len(comp) <= 25:
                path = find_longest_path_iterative(comp, local_adj)
                if len(path) > len(best_path):
                    best_path = path
            else:
                path = find_longest_path_dfs(comp, local_adj)
                if len(path) > len(best_path):
                    best_path = path
            
            # Also try tree diameter as a baseline
            tree_path = find_longest_path_in_tree(comp, local_adj)
            if len(tree_path) > len(best_path):
                best_path = tree_path
    
    # Final check: try to extend the best path
    if best_path and len(best_path) < n:
        # Try extending from both ends
        path_set = set(best_path)
        
        def try_extend(end_vertex, path):
            for v in adj[end_vertex]:
                if v not in path_set:
                    new_path = [v] + path if end_vertex == path[0] else path + [v]
                    return new_path
            return None
        
        # Try extending forward
        extended = try_extend(best_path[0], best_path)
        if extended:
            best_path = extended
        
        # Try extending backward
        extended = try_extend(best_path[-1], best_path)
        if extended:
            best_path = extended
    
    if not best_path:
        # Fallback: return a single vertex or simple edge
        if P['edges']:
            best_path = [P['edges'][0][0], P['edges'][0][1]]
        else:
            best_path = [0]
            
    return best_path
# EVOLVE-BLOCK-END
