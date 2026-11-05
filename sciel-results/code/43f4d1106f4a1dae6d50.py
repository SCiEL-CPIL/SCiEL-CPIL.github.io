def construct():
    n = 100
    edges = [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]]
    
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Find largest connected component
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
            
    largest_comp = max(components, key=len)
    comp_set = set(largest_comp)
    comp_adj = {v: adj[v] & comp_set for v in largest_comp}
    
    # Heuristic 1: Greedy construction with Warnsdorff's rule
    def greedy_path(start):
        path = [start]
        seen = {start}
        while True:
            curr = path[-1]
            candidates = [v for v in comp_adj[curr] if v not in seen]
            if not candidates:
                break
            # Choose neighbor with fewest unvisited neighbors
            v = min(candidates, key=lambda x: len(comp_adj[x] - seen))
            path.append(v)
            seen.add(v)
        return path

    best_path = []
    
    # Try greedy from each node in the component
    for start in largest_comp:
        p = greedy_path(start)
        if len(p) > len(best_path):
            best_path = p
            
    # Heuristic 2: Local Search - Insertion of unused vertices
    # We try to insert an unused vertex u into the path between a and b if (a,u) and (u,b) are edges.
    # This increases path length by 1.
    
    def improve_path(path):
        used = set(path)
        # List of unused vertices in the component
        unused = [v for v in largest_comp if v not in used]
        
        # Precompute adjacency for fast lookup
        # We check if inserting u between path[i] and path[i+1] is valid
        # This requires edge (path[i], u) and edge (u, path[i+1])
        
        improved = True
        while improved:
            improved = False
            # Shuffle unused to avoid bias? Or just iterate.
            # To be deterministic and fast, we can just check all.
            for u in unused[:]: # Iterate over a copy
                if u in used:
                    continue
                # Try inserting u at any position i (between path[i] and path[i+1])
                # Also try prepending (before path[0]) or appending (after path[-1])
                # Prepending: need edge (u, path[0])
                if u in comp_adj[path[0]]:
                    # Check if we can just prepend? 
                    # Path becomes [u, path[0], ...]
                    # This is valid if u is not used.
                    # But wait, if we prepend, the new start is u.
                    # Is it always beneficial? Yes, length increases.
                    path.insert(0, u)
                    used.add(u)
                    unused.remove(u)
                    improved = True
                    break # Restart scan with new path
                    
                # Appending: need edge (path[-1], u)
                if u in comp_adj[path[-1]]:
                    path.append(u)
                    used.add(u)
                    unused.remove(u)
                    improved = True
                    break # Restart scan
                    
                # Inserting in middle: between path[i] and path[i+1]
                for i in range(len(path) - 1):
                    a = path[i]
                    b = path[i+1]
                    if u in comp_adj[a] and u in comp_adj[b]:
                        path.insert(i+1, u)
                        used.add(u)
                        unused.remove(u)
                        improved = True
                        break # Restart scan
                if improved:
                    break
                    
        return path

    # Apply improvement to the best greedy path
    best_path = improve_path(best_path)
    
    # It's possible that a slightly shorter greedy path leads to a much longer improved path.
    # So we should apply improvement to several of the best greedy paths.
    
    greedy_paths = []
    for start in largest_comp:
        p = greedy_path(start)
        greedy_paths.append(p)
        
    # Sort by length descending
    greedy_paths.sort(key=len, reverse=True)
    
    # Try improving the top 10 greedy paths
    for p in greedy_paths[:10]:
        improved = improve_path(p)
        if len(improved) > len(best_path):
            best_path = improved
            
    return best_path
