import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]]}

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
            comp = []
            stack = [i]
            visited_comp[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for neighbor in adj[node]:
                    if not visited_comp[neighbor]:
                        visited_comp[neighbor] = True
                        stack.append(neighbor)
            components.append(comp)
    
    # Sort components by size, largest first
    components.sort(key=len, reverse=True)
    
    best_path = []
    deadline = time.time() + 50.0  # Safety margin
    
    # We focus on the largest component (Giant Component)
    if not components:
        return [0]
        
    target_comp = components[0]
    target_set = set(target_comp)
    comp_size = len(target_comp)
    
    # Precompute degrees within the component
    deg_in_comp = [len(adj[i] & target_set) for i in range(n)]
    
    # Helper to count unvisited neighbors in the component
    def get_unvisited_neighbors(node, visited):
        return [nb for nb in adj[node] if nb in target_set and nb not in visited]
    
    # Check if moving to 'nb' from 'node' would isolate any unvisited node
    def would_isolate(node, nb, visited):
        """Check if after visiting nb, any unvisited node becomes disconnected from the rest."""
        new_visited = visited | {nb}
        unvisited = target_set - new_visited
        if not unvisited:
            return False
        
        # Check connectivity from nb to all unvisited nodes
        # Use BFS from nb, but only through unvisited nodes + nb
        visited_bfs = {nb}
        stack = [nb]
        while stack:
            curr = stack.pop()
            for neigh in adj[curr]:
                if neigh in target_set and neigh not in visited_bfs:
                    if neigh in new_visited or neigh in unvisited:
                        visited_bfs.add(neigh)
                        stack.append(neigh)
        
        # All unvisited nodes should be reachable from nb
        for u in unvisited:
            if u not in visited_bfs:
                return True
        return False
    
    # Composite heuristic for neighbor ordering
    def neighbor_heuristic(nb, current_visited):
        """Lower score = better candidate. Combines Warnsdorff + degree."""
        unvisited_nbrs = get_unvisited_neighbors(nb, current_visited)
        warn_score = len(unvisited_nbrs)
        # Prefer nodes with fewer unvisited neighbors (Warnsdorff)
        # Tie-break: prefer lower degree in component
        return (warn_score, deg_in_comp[nb])
    
    best_len = 0
    best_path_found = []
    
    # Try starting from each node in the component
    # Prioritize: low-degree nodes first (more likely to be endpoints of a long path)
    starts = sorted(target_comp, key=lambda x: deg_in_comp[x])
    
    for start in starts:
        if time.time() > deadline:
            break
            
        visited = {start}
        path = [start]
        
        # Iterative DFS with backtracking using explicit stack
        # Each stack entry: (node, neighbor_index, path_copy, visited_copy)
        # But copying is expensive. Instead, use recursion with careful time checks.
        
        sys.setrecursionlimit(10000)
        
        def dfs(node, visited, path):
            nonlocal best_len, best_path_found
            
            if len(path) > best_len:
                best_len = len(path)
                best_path_found = path[:]
                
            # If we have visited all nodes in the component, we are done
            if len(visited) == comp_size:
                return True  # Found Hamiltonian path
            
            if time.time() > deadline:
                return False  # Time out
            
            neighbors = get_unvisited_neighbors(node, visited)
            
            if not neighbors:
                return False  # Dead end
            
            # Sort neighbors using composite heuristic
            # Warnsdorff: fewest unvisited neighbors first
            # Tie-break: lower degree in component
            neighbors.sort(key=lambda x: neighbor_heuristic(x, visited))
            
            for nb in neighbors:
                # Optional: check if this move would isolate nodes (expensive, use sparingly)
                # Only do this check if we have plenty of time
                if time.time() < deadline - 10:
                    if would_isolate(node, nb, visited):
                        continue
                
                visited.add(nb)
                path.append(nb)
                
                if dfs(nb, visited, path):
                    return True
                
                path.pop()
                visited.remove(nb)
                
                # If time is running out, return early
                if time.time() > deadline:
                    return False
                    
            return False
        
        dfs(start, visited, path)
        
        if best_len == comp_size:
            break  # Found Hamiltonian path, can't do better
            
    if best_path_found:
        return best_path_found
    
    # Fallback: If DFS didn't find anything good, return a simple greedy path
    if not best_path:
        # Greedy: Start from node with min degree, always go to neighbor with min unvisited count
        visited = set()
        current = starts[0]
        path = [current]
        visited.add(current)
        
        while True:
            if time.time() > deadline:
                break
            neighbors = [nb for nb in adj[current] if nb in target_set and nb not in visited]
            if not neighbors:
                break
            # Pick neighbor with smallest number of unvisited neighbors (Warnsdorff)
            neighbors.sort(key=lambda x: len(get_unvisited_neighbors(x, visited)))
            current = neighbors[0]
            path.append(current)
            visited.add(current)
             
        if path:
            return path
             
    return [0]  # Should not reach here if graph is non-empty

# EVOLVE-BLOCK-END
