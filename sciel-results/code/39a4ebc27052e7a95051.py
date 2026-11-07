import sys
import time

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    n = P['n']
    adj = {i: [] for i in range(n)}
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    
    forbidden_edges = set()
    
    def find_cycle():
        start = 0
        path = [start]
        visited = {start}
        
        def dfs(u):
            if time.time() > deadline:
                return False
                
            if len(path) == n:
                edge = tuple(sorted((u, start)))
                if edge not in forbidden_edges:
                    return True
                return False
            
            for v in adj[u]:
                if v in visited:
                    continue
                edge = tuple(sorted((u, v)))
                if edge in forbidden_edges:
                    continue
                
                visited.add(v)
                path.append(v)
                
                if dfs(v):
                    return True
                
                visited.remove(v)
                path.pop()
            
            return False
        
        if dfs(start):
            return path[:]
        return None

    cycles = []
    max_cycles = 3
    
    for _ in range(max_cycles):
        if time.time() > deadline:
            break
        cycle = find_cycle()
        if cycle is None:
            break
        cycles.append(cycle)
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            forbidden_edges.add(tuple(sorted((u, v))))

    if not cycles:
        return [list(range(n))]
        
    return cycles
# EVOLVE-BLOCK-END
