import time
import random

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin for 60s limit
    
    n = P['n']
    edges = set()
    for e in P['edges']:
        edges.add((min(e[0], e[1]), max(e[0], e[1])))
    
    # Target: Find 3 edge-disjoint Hamiltonian cycles
    target_cycles = 3
    
    def find_hamiltonian_cycle(adj, n, time_limit):
        """
        Finds a Hamiltonian cycle in the graph defined by adj.
        Returns a list of vertices or None.
        """
        # Check if graph is connected and all degrees >= 2
        # Quick check: if any node has degree < 2, no ham cycle
        for i in range(n):
            if len(adj[i]) < 2:
                return None
        
        # Start from node 0
        visited = [False] * n
        path = [0]
        visited[0] = True
        
        def dfs(v, depth):
            if time.time() > time_limit:
                return None
            
            if depth == n:
                # Check if we can close the cycle
                if 0 in adj[v]:
                    return path[:]
                return None
            
            # Get unvisited neighbors
            neighbors = [u for u in adj[v] if not visited[u]]
            
            # Warnsdorff's rule: sort by number of unvisited neighbors
            # This heuristic is very effective for Hamiltonian paths/cycles
            def sort_key(u):
                cnt = 0
                for w in adj[u]:
                    if not visited[w]:
                        cnt += 1
                return cnt
            
            neighbors.sort(key=sort_key)
            
            for u in neighbors:
                visited[u] = True
                path.append(u)
                
                res = dfs(u, depth + 1)
                if res:
                    return res
                
                path.pop()
                visited[u] = False
            
            return None
        
        return dfs(0, 1)

    def get_adjacency(edge_set):
        adj = {i: set() for i in range(n)}
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    # We will try to find 3 cycles.
    # To ensure we don't pick a "bad" first cycle that makes the second impossible,
    # we can use a limited backtracking strategy or multiple random restarts.
    # Given the small size (21 nodes), a single well-guided search often works.
    # However, to be robust, we'll try a few different random seeds/orders if the first attempt fails.
    
    best_result = []
    
    # Try a few different strategies/randomizations
    for attempt in range(20):
        if time.time() > deadline:
            break
            
        current_edges = edges.copy()
        cycles = []
        success = True
        
        for i in range(target_cycles):
            adj = get_adjacency(current_edges)
            
            # Find a Hamiltonian cycle
            # We can randomize the start node or neighbor order slightly if needed,
            # but Warnsdorff from node 0 is usually strong.
            # To diversify, we might shuffle the initial neighbor list if we get stuck,
            # but let's try deterministic first. If it fails, we restart.
            
            c = find_hamiltonian_cycle(adj, n, deadline)
            
            if c is None:
                success = False
                break
            else:
                cycles.append(c)
                # Remove edges of c from current_edges
                for j in range(len(c)):
                    u = c[j]
                    v = c[(j+1) % len(c)]
                    e = (min(u, v), max(u, v))
                    if e in current_edges:
                        current_edges.remove(e)
        
        if success and len(cycles) > len(best_result):
            best_result = cycles
            if len(best_result) == target_cycles:
                break # Found optimal
        
        # If we didn't find 3, we might want to try a different randomization
        # But the current heuristic is deterministic. 
        # To add randomness, we could shuffle the edges in the adjacency list construction
        # or the neighbor sorting ties.
        
    # Convert to required format
    result = []
    for c in best_result:
        result.append(list(c))
        
    return result

