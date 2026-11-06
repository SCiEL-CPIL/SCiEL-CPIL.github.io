"""Rewritten program: find a Hamilton cycle minimizing max edge arrival index."""
import time

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

def construct():
    n = P['n']
    edges = P['edge_order']
    # Build edge position map: (min(u,v), max(u,v)) -> 1-based index
    edge_pos = {}
    for i, (u, v) in enumerate(edges):
        a, b = min(u, v), max(u, v)
        edge_pos[(a, b)] = i + 1
    
    def score(cycle):
        """Return the max edge position for a given cycle (list of vertices)."""
        max_pos = 0
        for i in range(n):
            u = cycle[i]
            v = cycle[(i + 1) % n]
            a, b = min(u, v), max(u, v)
            pos = edge_pos[(a, b)]
            if pos > max_pos:
                max_pos = pos
        return max_pos
    
    # Try to find a Hamilton cycle using only edges with position <= K
    # We do this via backtracking with strong pruning.
    # Start with small K and increase.
    
    # Precompute adjacency for each K incrementally
    # But K can go up to ~276, so let's do it efficiently.
    
    # For a given K, build adjacency list
    def build_adj(K):
        adj = [set() for _ in range(n)]
        for i in range(K):
            u, v = edges[i]
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def has_ham_cycle(adj, K):
        """Check if there's a Hamilton cycle using only edges with position <= K.
        Returns the cycle as a list of vertices, or None."""
        # Use backtracking from vertex 0
        visited = [False] * n
        path = [0]
        visited[0] = True
        
        # Degree pruning: if any unvisited vertex has degree < 2 in the remaining graph, prune
        # Actually, for a Hamilton cycle, every vertex must have degree >= 2 in the cycle,
        # so in the subgraph, every vertex must have degree >= 2.
        
        # Check connectivity and min degree first
        deg = [len(adj[i]) for i in range(n)]
        for i in range(n):
            if deg[i] < 2:
                return None
        
        # Check if graph is connected
        from collections import deque
        seen = [False] * n
        dq = deque([0])
        seen[0] = True
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    dq.append(v)
        if not all(seen):
            return None
        
        # Backtracking with Warnsdorff's heuristic
        # Sort neighbors by degree (fewest first)
        
        def dfs(current, depth):
            if depth == n:
                # Check if current is connected to 0
                if 0 in adj[current]:
                    return list(path)
                return None
            
            # Pruning: for each unvisited vertex, check if it still has at least 2 unvisited neighbors
            # (one to enter, one to leave), except the last vertex which only needs 1 to connect back to 0
            for i in range(n):
                if not visited[i]:
                    unvis_deg = 0
                    for nb in adj[i]:
                        if not visited[nb]:
                            unvis_deg += 1
                    # The last vertex (depth == n-1) only needs to connect to 0
                    if depth == n - 1:
                        if unvis_deg < 1 and 0 not in adj[i]:
                            return None
                    else:
                        if unvis_deg < 2:
                            return None
            
            # Get unvisited neighbors
            neighbors = []
            for v in adj[current]:
                if not visited[v]:
                    neighbors.append(v)
            
            if not neighbors:
                return None
            
            # Warnsdorff: sort by number of unvisited neighbors (fewest first)
            def warnsdorff_key(v):
                cnt = 0
                for nb in adj[v]:
                    if not visited[nb] and nb != current:
                        cnt += 1
                return cnt
            
            neighbors.sort(key=warnsdorff_key)
            
            for v in neighbors:
                visited[v] = True
                path.append(v)
                result = dfs(v, depth + 1)
                if result is not None:
                    return result
                path.pop()
                visited[v] = False
            
            return None
        
        result = dfs(0, 1)
        return result
    
    # Try increasing K
    best_cycle = list(range(n))
    best_score = score(best_cycle)
    
    deadline = time.time() + 85  # 5 second safety margin
    
    # Start with a reasonable lower bound. The minimum possible max edge position
    # is at least 24 (need at least 24 edges for a cycle on 24 vertices).
    # But we need a connected graph, so start higher.
    # Let's start from K=30 and go up.
    
    for K in range(24, 277):
        if time.time() > deadline:
            break
        
        adj = build_adj(K)
        cycle = has_ham_cycle(adj, K)
        if cycle is not None:
            s = score(cycle)
            if s < best_score:
                best_score = s
                best_cycle = cycle
            # We found a cycle with all edges <= K, so score <= K
            # We can break since we're trying in increasing K order
            # But score might be less than K, so let's just return
            return best_cycle
    
    return best_cycle
# EVOLVE-BLOCK-END
