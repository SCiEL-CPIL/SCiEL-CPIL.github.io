"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

def construct():
    n = P['n']
    edge_order = P['edge_order']
    total_edges = len(edge_order)
    
    # Build edge index lookup
    edge_idx = {}
    for i, (u, v) in enumerate(edge_order):
        edge_idx[(min(u,v), max(u,v))] = i
    
    # Build adjacency list with edge indices
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edge_order):
        adj[u].append((v, i))
        adj[v].append((u, i))
    
    # Sort adjacency by edge index
    for v in range(n):
        adj[v].sort(key=lambda x: x[1])
    
    start_time = time.time()
    deadline = start_time + 85
    
    best_score = total_edges
    best_cycle = list(range(n))
    
    # DFS to find Hamilton cycle with edge limit
    def search(start, limit, time_limit):
        visited = [False] * n
        visited[start] = True
        path = [start]
        found = [False]
        result = [None]
        
        def dfs(current, depth):
            if found[0] or time.time() > time_limit:
                return
            if depth == n:
                for (nxt, eidx) in adj[current]:
                    if nxt == start and eidx <= limit:
                        found[0] = True
                        result[0] = list(path)
                        return
                return
            
            for (nxt, eidx) in adj[current]:
                if found[0] or time.time() > time_limit:
                    return
                if not visited[nxt] and eidx <= limit:
                    visited[nxt] = True
                    path.append(nxt)
                    dfs(nxt, depth + 1)
                    path.pop()
                    visited[nxt] = False
                    if found[0]:
                        return
        
        dfs(start, 1)
        return result[0]
    
    # Try to find the minimum limit K
    # Start from a high limit and decrease
    # First, ensure we have some solution
    for start in range(n):
        if time.time() > deadline - 10:
            break
        res = search(start, total_edges - 1, deadline - 10)
        if res is not None:
            best_cycle = res
            # Compute actual score
            max_eidx = 0
            for i in range(n):
                u = res[i]
                v = res[(i+1) % n]
                key = (min(u,v), max(u,v))
                max_eidx = max(max_eidx, edge_idx[key])
            best_score = max_eidx + 1
            break
    
    # Now try to improve by decreasing the limit
    current_limit = best_score - 1
    while current_limit >= 0:
        if time.time() > deadline - 5:
            break
        found_any = False
        for start in range(n):
            if time.time() > deadline - 5:
                break
            res = search(start, current_limit, deadline - 5)
            if res is not None:
                found_any = True
                # Verify and compute score
                max_eidx = 0
                for i in range(n):
                    u = res[i]
                    v = res[(i+1) % n]
                    key = (min(u,v), max(u,v))
                    max_eidx = max(max_eidx, edge_idx[key])
                actual_score = max_eidx + 1
                if actual_score < best_score:
                    best_score = actual_score
                    best_cycle = res
                break
        
        if found_any:
            current_limit = best_score - 1
        else:
            current_limit -= 1
    
    return best_cycle

# EVOLVE-BLOCK-END
