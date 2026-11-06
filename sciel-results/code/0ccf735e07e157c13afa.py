import sys
import time
from collections import defaultdict, deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

def construct():
    n = P['n']
    edges = P['edge_order']
    num_edges = len(edges)
    
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Map edge to its 1-based index
    edge_to_idx = {}
    for i, (u, v) in enumerate(edges):
        if u > v:
            u, v = v, u
        edge_to_idx[(u, v)] = i + 1
    
    def build_adjacency(K):
        """Build adjacency list and degree array for first K edges."""
        adj = [set() for _ in range(n)]
        for i in range(K):
            u, v = edges[i]
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def check_basic_feasible(adj):
        """Quick feasibility: min degree >= 2, connectivity."""
        for i in range(n):
            if len(adj[i]) < 2:
                return False
        visited = [False] * n
        visited[0] = True
        stack = [0]
        count = 0
        while stack:
            node = stack.pop()
            count += 1
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        return count == n
    
    def hopcroft_karp(adj):
        """Check if perfect matching exists in bipartite graph (left=exits, right=entries)."""
        left_adj = [[] for _ in range(n)]
        for u in range(n):
            for v in adj[u]:
                if u != v:
                    left_adj[u].append(v)
        
        for u in range(n):
            if len(left_adj[u]) < 1:
                return False
        
        pair_u = [-1] * n
        pair_v = [-1] * n
        dist = [0] * n
        
        def bfs():
            queue = deque()
            for u in range(n):
                if pair_u[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = float('inf')
            found = False
            while queue:
                u = queue.popleft()
                for v in left_adj[u]:
                    if pair_v[v] == -1:
                        found = True
                    elif dist[pair_v[v]] == float('inf'):
                        dist[pair_v[v]] = dist[u] + 1
                        queue.append(pair_v[v])
            return found
        
        def dfs(u):
            for v in left_adj[u]:
                if pair_v[v] == -1 or (dist[pair_v[v]] == dist[u] + 1 and dfs(pair_v[v])):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = float('inf')
            return False
        
        matching = 0
        while bfs():
            for u in range(n):
                if pair_u[u] == -1:
                    if dfs(u):
                        matching += 1
        
        return matching == n
    
    def find_hamilton_cycle(K, adj=None):
        """Find Hamilton cycle in first K edges using multi-start backtracking."""
        if adj is None:
            adj = build_adjacency(K)
        
        if not check_basic_feasible(adj):
            return None
        
        if not hopcroft_karp(adj):
            return None
        
        adj_list = [list(s) for s in adj]
        
        # Try multiple starting vertices sorted by degree
        start_order = sorted(range(n), key=lambda x: len(adj_list[x]))
        
        for start in start_order:
            if time.time() >= deadline:
                return None
            
            path = [start]
            used = [False] * n
            used[start] = True
            unvisited_deg = [len(adj_list[i]) for i in range(n)]
            unvisited_deg[start] = 0
            
            def backtrack(current, depth):
                if time.time() >= deadline:
                    return False
                if depth == n:
                    return start in adj[current]
                
                unvisited_neighbors = [nb for nb in adj_list[current] if not used[nb]]
                if not unvisited_neighbors:
                    return False
                
                unvisited_neighbors.sort(key=lambda x: (unvisited_deg[x], len(adj_list[x])))
                
                for nb in unvisited_neighbors:
                    if depth == n - 2:
                        remaining = None
                        for v in range(n):
                            if not used[v] and v != nb:
                                remaining = v
                                break
                        if remaining is not None:
                            if nb not in adj[remaining] or start not in adj[remaining]:
                                continue
                    
                    path.append(nb)
                    used[nb] = True
                    unvisited_deg[nb] = 0
                    for neighbor in adj_list[nb]:
                        if not used[neighbor]:
                            unvisited_deg[neighbor] -= 1
                    
                    if backtrack(nb, depth + 1):
                        return True
                    
                    path.pop()
                    used[nb] = False
                    unvisited_deg[nb] = len(adj_list[nb])
                    for neighbor in adj_list[nb]:
                        if not used[neighbor]:
                            unvisited_deg[neighbor] += 1
                
                return False
            
            if backtrack(start, 1):
                cycle = path
                if 0 in cycle:
                    idx = cycle.index(0)
                    cycle = cycle[idx:] + cycle[:idx]
                return cycle
        
        return None
    
    # Hybrid search: coarse binary search then fine linear scan
    low = n
    high = num_edges
    best_cycle = None
    best_K = num_edges + 1
    
    # Coarse binary search
    while low <= high and time.time() < deadline:
        mid = (low + high) // 2
        adj_mid = build_adjacency(mid)
        cycle = find_hamilton_cycle(mid, adj_mid)
        if cycle is not None:
            best_cycle = cycle
            best_K = mid
            high = mid - 1
        else:
            low = mid + 1
    
    # Fine linear scan around the found range
    if best_cycle is not None:
        scan_low = max(n, best_K - 5)
        scan_high = best_K
        for K in range(scan_low, scan_high + 1):
            if time.time() >= deadline:
                break
            if K < best_K:
                adj_K = build_adjacency(K)
                cycle = find_hamilton_cycle(K, adj_K)
                if cycle is not None:
                    best_cycle = cycle
                    best_K = K
    
    if best_cycle is None:
        for K in range(n, num_edges + 1):
            if time.time() >= deadline:
                break
            adj_K = build_adjacency(K)
            cycle = find_hamilton_cycle(K, adj_K)
            if cycle is not None:
                best_cycle = cycle
                best_K = K
                break
    
    if best_cycle is None:
        cycle = find_hamilton_cycle(num_edges)
        if cycle is not None:
            best_cycle = cycle
        else:
            best_cycle = list(range(n))
    
    return best_cycle

# EVOLVE-BLOCK-END
