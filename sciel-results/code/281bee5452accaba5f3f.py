import time
import random

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Function to find a Hamiltonian cycle in the current graph (defined by adj)
    # Returns list of vertices or None
    def find_hc(adj):
        # Check degree constraints
        for v in range(n):
            if len(adj[v]) < 2:
                return None
        
        # DFS with heuristics
        visited = [False] * n
        path = [0]
        visited[0] = True
        
        def dfs(current, depth):
            if depth == n:
                # Check if we can close the cycle
                if 0 in adj[current]:
                    return True
                return False
            
            # Get unvisited neighbors
            unvisited = [v for v in adj[current] if not visited[v]]
            if not unvisited:
                return False
            
            # Heuristic: Sort by degree in remaining graph (Warnsdorff's rule)
            unvisited.sort(key=lambda v: len([x for x in adj[v] if not visited[x]]))
            
            for nxt in unvisited:
                visited[nxt] = True
                path.append(nxt)
                if dfs(nxt, depth + 1):
                    return True
                path.pop()
                visited[nxt] = False
            
            return False
        
        if dfs(0, 1):
            return list(path)
        return None

    cycles = []
    current_adj = [set(s) for s in adj]
    
    # Try to find 3 edge-disjoint Hamiltonian cycles
    for i in range(3):
        hc = find_hc(current_adj)
        if hc is None:
            break
        
        cycles.append(hc)
        
        # Remove edges of this cycle from current_adj
        for j in range(len(hc)):
            u = hc[j]
            v = hc[(j + 1) % len(hc)]
            current_adj[u].remove(v)
            current_adj[v].remove(u)
            
    # If we didn't find 3, try randomized restarts for the remaining slots
    if len(cycles) < 3:
        for attempt in range(10):
            random.seed(attempt)
            current_adj = [set(s) for s in adj]
            cycles = []
            for i in range(3):
                # Note: find_hc is deterministic. 
                # For robustness, we could shuffle adjacency, but the graph is small.
                hc = find_hc(current_adj)
                if hc is None:
                    break
                cycles.append(hc)
                for j in range(len(hc)):
                    u = hc[j]
                    v = hc[(j + 1) % len(hc)]
                    current_adj[u].remove(v)
                    current_adj[v].remove(u)
            if len(cycles) == 3:
                break

    return [[int(v) for v in cycle] for cycle in cycles]
