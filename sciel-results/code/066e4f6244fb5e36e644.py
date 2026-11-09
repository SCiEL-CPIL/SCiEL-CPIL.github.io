import time

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    start_time = time.time()
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    adj_list = [list(s) for s in adj]
    
    used_edges = set()
    cycles = []
    
    # Deadline guard: 55 seconds
    deadline = start_time + 55.0

    def find_cycle(used_edges_set):
        visited = [False] * n
        path = [0]
        visited[0] = True
        
        def dfs(u):
            if time.time() > deadline:
                return False
                
            if len(path) == n:
                # Check if closing edge is available
                if (u, 0) not in used_edges_set and (0, u) not in used_edges_set:
                    return True
                return False
            
            # Generate candidate neighbors
            candidates = []
            for v in adj_list[u]:
                if not visited[v]:
                    # Check if edge (u, v) is available
                    if (u, v) not in used_edges_set and (v, u) not in used_edges_set:
                        # Heuristic: count available unvisited neighbors for v
                        # This helps prune branches that lead to dead ends sooner
                        cnt = 0
                        for w in adj_list[v]:
                            if not visited[w]:
                                if (v, w) not in used_edges_set and (w, v) not in used_edges_set:
                                    cnt += 1
                        candidates.append((cnt, v))
            
            # Sort by number of options (Warnsdorff's rule)
            candidates.sort()
            
            for cnt, v in candidates:
                visited[v] = True
                path.append(v)
                if dfs(v):
                    return True
                path.pop()
                visited[v] = False
            return False
        
        if dfs(0):
            return path
        return None

    # Try to find up to 3 edge-disjoint Hamiltonian cycles
    for _ in range(3):
        if time.time() > deadline:
            break
            
        cycle = find_cycle(used_edges)
        if cycle is None:
            break
            
        cycles.append(cycle)
        
        # Mark edges in this cycle as used
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            if u < v:
                used_edges.add((u, v))
            else:
                used_edges.add((v, u))
                
    return cycles
