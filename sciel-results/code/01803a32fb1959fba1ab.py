import time
from collections import deque
import random

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
    
    deg = {u: len([v for v in adj_list[u] if v in comp_set]) for u in largest_comp}
    
    e_count = sum(deg[u] for u in largest_comp) // 2
    is_tree = (e_count == comp_size - 1)
    
    best_path = []
    best_len = 0
    
    deadline = time.time() + 85.0
    
    if is_tree:
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
        best_path = path
        best_len = len(best_path)
    else:
        # Non-tree: use DFS with Warnsdorff heuristic and time budget
        
        def dfs_path(start, path, visited_set, max_depth=None):
            nonlocal best_path, best_len
            
            if time.time() > deadline:
                return False
            
            if len(path) > best_len:
                best_len = len(path)
                best_path = path[:]
            
            if len(path) == comp_size:
                return True
            
            if max_depth is not None and len(path) >= max_depth:
                return False
            
            curr = path[-1]
            remaining = comp_size - len(visited_set)
            
            if len(path) + remaining <= best_len:
                return False
            
            neighbors = [v for v in adj_list[curr] if v in comp_set and v not in visited_set]
            
            if not neighbors:
                return False
            
            # Warnsdorff: fewest onward moves first, then lowest degree
            def onward_count(v):
                count = 0
                for w in adj_list[v]:
                    if w in comp_set and w not in visited_set and w != curr:
                        count += 1
                return count
            
            neighbors.sort(key=lambda x: (onward_count(x), deg[x]))
            
            for v in neighbors:
                path.append(v)
                visited_set.add(v)
                
                if dfs_path(v, path, visited_set, max_depth):
                    path.pop()
                    visited_set.remove(v)
                    return True
                
                path.pop()
                visited_set.remove(v)
                
                if time.time() > deadline:
                    return False
            
            return False
        
        # Try starting from low-degree nodes (more constrained)
        start_nodes = sorted(largest_comp, key=lambda x: deg[x])
        
        for start in start_nodes[:20]:
            if time.time() > deadline:
                break
            dfs_path(start, [start], {start})
            if best_len == comp_size:
                break
        
        # If not Hamiltonian, try high-degree starts too
        if best_len < comp_size and time.time() < deadline:
            high_degree_starts = sorted(largest_comp, key=lambda x: -deg[x])
            for start in high_degree_starts[:10]:
                if time.time() > deadline:
                    break
                if dfs_path(start, [start], {start}):
                    if best_len == comp_size:
                        break
        
        # Local improvement: try to extend the best path from both ends
        if len(best_path) > 0 and time.time() < deadline:
            def extend_path(path, visited_set):
                """Try to extend path from both ends greedily."""
                extended = path[:]
                ext_visited = set(extended)
                
                # Extend from start
                while True:
                    curr = extended[0]
                    nexts = [v for v in adj_list[curr] if v in comp_set and v not in ext_visited]
                    if not nexts:
                        break
                    nexts.sort(key=lambda x: (len([w for w in adj_list[x] if w in comp_set and w not in ext_visited]), deg[x]))
                    extended.insert(0, nexts[0])
                    ext_visited.add(nexts[0])
                
                # Extend from end
                while True:
                    curr = extended[-1]
                    nexts = [v for v in adj_list[curr] if v in comp_set and v not in ext_visited]
                    if not nexts:
                        break
                    nexts.sort(key=lambda x: (len([w for w in adj_list[x] if w in comp_set and w not in ext_visited]), deg[x]))
                    extended.append(nexts[0])
                    ext_visited.add(nexts[0])
                
                return extended
            
            extended = extend_path(best_path, set(best_path))
            if len(extended) > best_len:
                best_path = extended
                best_len = len(extended)
        
        # Iterative local search: try swapping endpoints and re-extending
        if best_len < comp_size and time.time() < deadline:
            for _ in range(50):
                if time.time() > deadline:
                    break
                if not best_path or len(best_path) < 2:
                    break
                
                # Try removing an internal vertex and re-inserting at an end
                improved = False
                path_set = set(best_path)
                
                for i in range(1, len(best_path) - 1):
                    if time.time() > deadline:
                        break
                    
                    removed = best_path[i]
                    # Check if removing it keeps the path connected
                    prev_node = best_path[i-1]
                    next_node = best_path[i+1]
                    if prev_node in adj[next_node]:
                        # Remove and try to re-insert at an end
                        candidate = best_path[:i] + best_path[i+1:]
                        cand_set = set(candidate)
                        
                        # Try to add removed at start
                        if removed in adj_list[candidate[0]]:
                            new_path = [removed] + candidate
                            new_set = set(new_path)
                            ext = extend_path(new_path, new_set)
                            if len(ext) > best_len:
                                best_path = ext
                                best_len = len(ext)
                                improved = True
                                break
                        
                        # Try to add removed at end
                        if removed in adj_list[candidate[-1]]:
                            new_path = candidate + [removed]
                            new_set = set(new_path)
                            ext = extend_path(new_path, new_set)
                            if len(ext) > best_len:
                                best_path = ext
                                best_len = len(ext)
                                improved = True
                                break
                
                if not improved:
                    break
        
        if not best_path:
            best_path = [largest_comp[0]]
    
    return best_path
