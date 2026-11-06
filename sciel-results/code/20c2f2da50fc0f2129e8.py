"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Compute degree for each vertex
    degree = [len(adj[i]) for i in range(n)]
    
    # Sort vertices by degree (ascending) - low degree vertices are good path endpoints
    vertices_by_degree = sorted(range(n), key=lambda x: (degree[x], x))
    
    best_path = []
    best_len = 0
    deadline = time.time() + 55  # Leave 5s margin
    
    def dfs_warnsdorff(start, time_limit):
        """DFS with Warnsdorff heuristic: prefer neighbors with fewer unvisited options."""
        nonlocal best_path, best_len
        path = [start]
        visited = {start}
        
        def _dfs():
            nonlocal best_path, best_len
            if time.time() > time_limit:
                return False
            if len(path) > best_len:
                best_len = len(path)
                best_path = path[:]
            
            current = path[-1]
            # Get unvisited neighbors, sorted by Warnsdorff heuristic
            unvisited_neighbors = [v for v in adj[current] if v not in visited]
            if not unvisited_neighbors:
                return True  # Dead end, backtrack
            
            # Sort by number of unvisited neighbors (Warnsdorff: prefer fewer options)
            unvisited_neighbors.sort(key=lambda v: (len(adj[v] - visited), v))
            
            for v in unvisited_neighbors:
                if time.time() > time_limit:
                    return True
                path.append(v)
                visited.add(v)
                if not _dfs():
                    return True  # Time limit reached
                path.pop()
                visited.remove(v)
            return True
        
        _dfs()
        return best_path
    
    # Try starting from each vertex, prioritizing low-degree vertices
    for start in vertices_by_degree:
        if time.time() > deadline:
            break
        if degree[start] == 0:
            continue
        # Allocate more time for promising starts (lower degree)
        time_per_start = 0.5
        dfs_warnsdorff(start, time.time() + time_per_start)
    
    # If we still have time, do additional randomized restarts
    if time.time() < deadline - 1:
        rng = random.Random(42)
        attempts = 0
        while time.time() < deadline - 1 and attempts < 50:
            start = rng.randint(0, n - 1)
            if degree[start] == 0:
                attempts += 1
                continue
            time_per_attempt = 0.3
            dfs_warnsdorff(start, time.time() + time_per_attempt)
            attempts += 1
    
    # Fallback: if no path found (shouldn't happen), return a single vertex
    if not best_path:
        best_path = [0]
    
    return best_path
# EVOLVE-BLOCK-END
