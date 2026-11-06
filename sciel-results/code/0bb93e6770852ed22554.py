import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Sort start vertices by degree (ascending) to likely start at endpoints
    degrees = [len(adj[i]) for i in range(n)]
    start_order = sorted(range(n), key=lambda x: (degrees[x], x))
    
    best_path = []
    best_len = 0
    start_time = time.time()
    deadline = start_time + 85.0 # 5s margin
    
    def dfs(path, visited):
        nonlocal best_path, best_len, deadline, start_time
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
            if best_len == n:
                return True # Found Hamiltonian path
        
        # Time check
        if time.time() > deadline:
            return False
        
        current = path[-1]
        neighbors = [v for v in adj[current] if not visited[v]]
        
        if not neighbors:
            return False
        
        # Warnsdorff Rule: Sort by number of unvisited neighbors
        # We want to visit vertices with fewer unvisited neighbors first
        def heuristic(v):
            unvisited_count = sum(1 for u in adj[v] if not visited[u])
            return (unvisited_count, v)
        
        neighbors.sort(key=heuristic)
        
        for v in neighbors:
            path.append(v)
            visited[v] = True
            if dfs(path, visited):
                return True
            path.pop()
            visited[v] = False
            
            # If time is running out, stop
            if time.time() > deadline:
                return False
                
        return False

    for start in start_order:
        if time.time() > deadline:
            break
            
        # Optional: Skip if starting vertex degree is 0
        if degrees[start] == 0:
            continue
            
        path = [start]
        visited = [False] * n
        visited[start] = True
        
        dfs(path, visited)
        
        # If we found a Hamiltonian path, we can stop
        if best_len == n:
            break

    return best_path
# EVOLVE-BLOCK-END
