import time

def construct():
    P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    def find_ham(start):
        path = [start]
        visited = {start}
        
        def backtrack():
            if len(path) == n:
                if start in adj[path[-1]]:
                    return True
                return False
            
            current = path[-1]
            # Heuristic: sort neighbors by degree in remaining graph? 
            # For simplicity, just try all.
            # To improve speed, we can check if unvisited neighbors have at least 1 unvisited neighbor (except the last one).
            
            for neighbor in list(adj[current]):
                if neighbor not in visited:
                    # Check if neighbor is a dead end (unless it's the last step)
                    if len(path) < n - 1:
                        unvisited_neighbors = [nb for nb in adj[neighbor] if nb not in visited]
                        if not unvisited_neighbors:
                            continue
                    
                    path.append(neighbor)
                    visited.add(neighbor)
                    
                    if backtrack():
                        return True
                        
                    path.pop()
                    visited.remove(neighbor)
            
            return False
            
        if backtrack():
            return path
        return None

    # Try to find first cycle
    # Start from 0
    c1 = find_ham(0)
    if not c1:
        return []
        
    # Remove edges of c1
    adj2 = {i: set(adj[i]) for i in range(n)}
    for i in range(len(c1)):
        u = c1[i]
        v = c1[(i+1) % len(c1)]
        adj2[u].remove(v)
        adj2[v].remove(u)
        
    # Check if adj2 has min degree >= 2
    min_deg = min(len(nei) for nei in adj2.values())
    if min_deg < 2:
        return [c1]
        
    # Try to find second cycle in adj2
    # We need to modify find_ham to use adj2
    def find_ham_adj(adj_graph, start):
        path = [start]
        visited = {start}
        
        def backtrack():
            if len(path) == n:
                if start in adj_graph[path[-1]]:
                    return True
                return False
            
            current = path[-1]
            for neighbor in list(adj_graph[current]):
                if neighbor not in visited:
                    if len(path) < n - 1:
                        unvisited_neighbors = [nb for nb in adj_graph[neighbor] if nb not in visited]
                        if not unvisited_neighbors:
                            continue
                    
                    path.append(neighbor)
                    visited.add(neighbor)
                    
                    if backtrack():
                        return True
                        
                    path.pop()
                    visited.remove(neighbor)
            
            return False
            
        if backtrack():
            return path
        return None

    c2 = find_ham_adj(adj2, 0)
    if c2:
        return [c1, c2]
    else:
        # If c2 not found, maybe try a different c1?
        # For now, return c1
        return [c1]
