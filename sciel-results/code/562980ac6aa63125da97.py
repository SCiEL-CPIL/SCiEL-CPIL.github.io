import sys
import time

# Increase recursion limit for DFS depth up to 100
sys.setrecursionlimit(200)

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Pre-compute degrees for sorting starting nodes
    degrees = [len(adj[i]) for i in range(n)]
    
    best_path = []
    start_time = time.time()
    timeout = 85.0  # Safety margin before 90s kill
    
    # Sort starting nodes by degree (ascending) to prioritize constrained nodes
    start_nodes = sorted(range(n), key=lambda x: degrees[x])
    
    # Global state for DFS
    path = []
    visited = [False] * n
    
    def dfs(current, depth):
        nonlocal best_path, start_time, timeout
        
        # Check time limit
        if time.time() - start_time > timeout:
            return False
        
        # Update best path
        if depth > len(best_path):
            best_path = path[:]
        
        # If we've visited all nodes, we found a Hamiltonian path
        if depth == n:
            return True
            
        # Get unvisited neighbors
        neighbors = [nb for nb in adj[current] if not visited[nb]]
        
        if not neighbors:
            return False
        
        # Warnsdorff's rule: sort neighbors by number of unvisited neighbors (ascending)
        # This prioritizes moves that leave fewer options for the next step,
        # helping to avoid getting stuck in a dead end.
        def warnsdorff_key(nb):
            return len([x for x in adj[nb] if not visited[x]])
        
        neighbors.sort(key=warnsdorff_key)
        
        for nb in neighbors:
            visited[nb] = True
            path.append(nb)
            
            if dfs(nb, depth + 1):
                return True
            
            path.pop()
            visited[nb] = False
            
            # Check time limit inside loop as well
            if time.time() - start_time > timeout:
                return False
                
        return False

    # Try starting from each node, prioritizing low-degree nodes
    for start in start_nodes:
        if time.time() - start_time > timeout:
            break
            
        # Reset state
        path = [start]
        visited = [False] * n
        visited[start] = True
        
        # If we already have a path of length n, stop
        if len(best_path) == n:
            break
            
        dfs(start, 1)
        
        # If we found a Hamiltonian path, we can stop
        if len(best_path) == n:
            break

    # Fallback: if DFS didn't find anything (unlikely), return a single node or simple greedy
    if not best_path:
        # Simple greedy fallback
        path = [0]
        seen = {0}
        while adj[path[-1]] - seen:
            v = min(adj[path[-1]] - seen, key=lambda w: len(adj[w] - seen))
            path.append(v)
            seen.add(v)
        best_path = path

    return best_path
