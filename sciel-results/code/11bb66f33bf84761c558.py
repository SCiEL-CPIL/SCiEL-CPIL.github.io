import sys
import time
import random
import collections

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
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
                for nei in adj[node]:
                    if not visited[nei]:
                        visited[nei] = True
                        stack.append(nei)
            components.append(comp)
            
    # Sort components by size, largest first
    components.sort(key=len, reverse=True)
    
    best_path = []
    deadline = time.time() + 85.0 # Safety margin
    
    def is_valid_path(path):
        if len(path) < 1:
            return False
        if len(set(path)) != len(path):
            return False
        for i in range(len(path) - 1):
            if path[i+1] not in adj[path[i]]:
                return False
        return True
    
    def bfs_farthest(start, comp_set):
        dist = {start: 0}
        q = collections.deque([start])
        farthest = start
        while q:
            node = q.popleft()
            for nei in adj[node]:
                if nei in comp_set and nei not in dist:
                    dist[nei] = dist[node] + 1
                    q.append(nei)
                    if dist[nei] > dist[farthest]:
                        farthest = nei
        return farthest, dist
    
    def bfs_path(start, target, comp_set):
        prev = {start: None}
        q = collections.deque([start])
        while q:
            node = q.popleft()
            if node == target:
                break
            for nei in adj[node]:
                if nei in comp_set and nei not in prev:
                    prev[nei] = node
                    q.append(nei)
        path = []
        curr = target
        while curr is not None:
            path.append(curr)
            curr = prev[curr]
        return path[::-1]
    
    def dfs_search(start, comp_set, comp_size):
        nonlocal best_path
        if time.time() > deadline:
            return
        
        visited_mask = {start}
        path = [start]
        
        # Use iterative DFS with explicit stack to avoid recursion limits
        # Stack entries: (node, visited_mask, path)
        # But copying visited_mask is expensive. Let's use recursive with sys.setrecursionlimit
        # For N=100, recursion depth is at most 100, which is fine.
        
        def _dfs(node, visited_mask, path):
            nonlocal best_path
            if time.time() > deadline:
                return
            if len(path) > len(best_path):
                best_path = path[:]
            
            # Pruning
            if len(path) + (comp_size - len(visited_mask)) <= len(best_path):
                return
                
            # Get unvisited neighbors
            unvisited_neis = [nei for nei in adj[node] if nei in comp_set and nei not in visited_mask]
            
            if not unvisited_neis:
                return
                
            # Sort by degree (descending) to prioritize "open" nodes
            unvisited_neis.sort(key=lambda x: len(adj[x]), reverse=True)
            
            for nei in unvisited_neis:
                if time.time() > deadline:
                    return
                visited_mask.add(nei)
                path.append(nei)
                _dfs(nei, visited_mask, path)
                path.pop()
                visited_mask.remove(nei)
        
        _dfs(start, visited_mask, path)
    
    def two_opt_improve(path, comp_set):
        """Try to extend or improve the path using 2-opt style moves."""
        nonlocal best_path
        if time.time() > deadline:
            return
        
        path = list(path)
        improved = True
        max_iterations = 200  # Limit iterations
        
        while improved and max_iterations > 0:
            improved = False
            max_iterations -= 1
            
            if time.time() > deadline:
                break
            
            # Try inserting each unvisited node into the path
            visited_in_path = set(path)
            unvisited = [v for v in comp_set if v not in visited_in_path]
            
            # Sort unvisited by degree for better ordering
            unvisited.sort(key=lambda x: len(adj[x]), reverse=True)
            
            for u in unvisited:
                if time.time() > deadline:
                    break
                    
                # Try inserting u between path[i] and path[i+1]
                for i in range(len(path) - 1):
                    if path[i] in adj[u] and path[i+1] in adj[u]:
                        new_path = path[:i+1] + [u] + path[i+1:]
                        if len(new_path) > len(best_path):
                            best_path = new_path[:]
                            path = new_path
                            improved = True
                            break
                else:
                    continue
                break
            
            # Try reversing a segment
            for i in range(len(path)):
                for j in range(i + 2, len(path)):
                    if time.time() > deadline:
                        break
                    # Check if reversing path[i:j+1] creates a valid path
                    # path[:i] + reversed(path[i:j+1]) + path[j+1:]
                    # Need edge from path[i-1] to path[j] if i > 0
                    # Need edge from path[i] to path[j+1] if j < len(path)-1
                    left_ok = True
                    right_ok = True
                    if i > 0:
                        if path[i-1] not in adj[path[j]]:
                            left_ok = False
                    if j < len(path) - 1:
                        if path[i] not in adj[path[j+1]]:
                            right_ok = False
                            
                    if left_ok and right_ok:
                        new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
                        if len(new_path) > len(best_path):
                            best_path = new_path[:]
                            path = new_path
                            improved = True
                            break
                if improved:
                    break
            if time.time() > deadline:
                break
        
        return path
    
    # Process components
    for comp in components:
        if time.time() > deadline:
            break
            
        comp_set = set(comp)
        comp_size = len(comp)
        
        # Count edges in component
        edge_count = 0
        for u in comp:
            edge_count += len(adj[u] & comp_set)
        edge_count //= 2
        
        if edge_count == comp_size - 1:
            # It's a tree. Find diameter.
            u, _ = bfs_farthest(comp[0], comp_set)
            v, _ = bfs_farthest(u, comp_set)
            path = bfs_path(u, v, comp_set)
            if len(path) > len(best_path):
                best_path = path[:]
            continue
        
        # For non-tree components, run DFS from selected start nodes
        # Use degree-based heuristic for start node selection
        # Try low-degree nodes first (they're more likely to be endpoints of long paths)
        nodes_by_degree = sorted(comp, key=lambda x: len(adj[x]))
        
        # Try all nodes if component is small, otherwise a subset
        if comp_size <= 30:
            starts = nodes_by_degree
        else:
            # Try a mix: low degree, high degree, and some random
            starts = nodes_by_degree[:10] + nodes_by_degree[-10:]
            # Add some random nodes
            import random as _rand
            _rand.seed(42)
            for _ in range(10):
                starts.append(_rand.choice(comp))
        
        for start in starts:
            if time.time() > deadline:
                break
            dfs_search(start, comp_set, comp_size)
        
        # Apply 2-opt improvement to the best path found
        if best_path:
            best_path = two_opt_improve(best_path, comp_set)
        
    # Final validation and return
    if not best_path:
        if edges:
            return [edges[0][0], edges[0][1]]
        else:
            return [0]
            
    if not is_valid_path(best_path):
        # Fallback to a valid path
        for u, v in edges:
            return [u, v]
        return [0]
        
    return best_path
# EVOLVE-BLOCK-END
