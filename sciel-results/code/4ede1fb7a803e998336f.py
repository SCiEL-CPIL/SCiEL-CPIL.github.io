import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = 24
    edge_order = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

    # Precompute adjacency incrementally
    adj = [set() for _ in range(n)]

    # Exact Hamilton cycle search with strong pruning
    def find_hamilton_cycle(adj, time_limit):
        """
        Find a Hamilton cycle in the graph defined by adj using exact backtracking
        with degree-based pruning. Returns the path as a list of vertices, or None.
        """
        t0 = time.time()
        
        # Check: all vertices must have degree >= 2
        for v in range(n):
            if len(adj[v]) < 2:
                return None
        
        # Check connectivity
        visited = [False] * n
        stack = [0]
        visited[0] = True
        cnt = 1
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    cnt += 1
                    stack.append(v)
        if cnt < n:
            return None
        
        # Use bitmask for visited set (n=24 fits in int)
        all_mask = (1 << n) - 1
        
        def dfs(path, current, visited_mask, start):
            if time.time() - t0 > time_limit:
                return None
            
            if len(path) == n:
                if start in adj[current]:
                    return list(path)
                return None
            
            # Pruning: check if any unvisited vertex has degree < 2 in the 
            # subgraph induced by unvisited vertices + current + start
            # (since it needs 2 edges in the final cycle)
            unvisited = [v for v in range(n) if not (visited_mask >> v) & 1]
            if unvisited:
                for v in unvisited:
                    # Count neighbors of v that are either in path (only current and start matter for closure)
                    # or unvisited
                    deg_in_reduced = 0
                    for w in adj[v]:
                        if (visited_mask >> w) & 1:
                            # w is visited; only current or start can connect
                            if w == current or w == start:
                                deg_in_reduced += 1
                        else:
                            deg_in_reduced += 1
                    if deg_in_reduced < 2:
                        return None
            
            # Get unvisited neighbors of current
            candidates = [v for v in adj[current] if not (visited_mask >> v) & 1]
            if not candidates:
                return None
            
            # Warnsdorff ordering: sort by number of unvisited neighbors
            def unvisited_deg(v):
                cnt = 0
                for w in adj[v]:
                    if not (visited_mask >> w) & 1:
                        cnt += 1
                return cnt
            
            candidates.sort(key=lambda v: unvisited_deg(v))
            
            for v in candidates:
                new_mask = visited_mask | (1 << v)
                result = dfs(path + [v], v, new_mask, start)
                if result is not None:
                    return result
            
            return None
        
        # Try starting from vertex 0
        for start in range(n):
            if time.time() - t0 > time_limit:
                break
            result = dfs([start], start, (1 << start), start)
            if result is not None:
                # Rotate so 0 is first
                idx0 = result.index(0)
                return result[idx0:] + result[:idx0]
        
        return None

    best_result = None
    
    # Incrementally add edges and check for Hamilton cycle
    for K in range(1, len(edge_order) + 1):
        if time.time() > deadline:
            break
        
        u, v = edge_order[K - 1]
        adj[u].add(v)
        adj[v].add(u)
        
        # Only check when all vertices have degree >= 2
        if all(len(adj[i]) >= 2 for i in range(n)):
            # Give more time for larger K where cycles are more likely
            time_remaining = deadline - time.time()
            if time_remaining < 1.0:
                break
            time_alloc = min(2.0, time_remaining * 0.5)
            
            result = find_hamilton_cycle(adj, time_alloc)
            if result is not None:
                best_result = result
                break
    
    if best_result is not None:
        return best_result
    
    return list(range(n))

print(construct())
