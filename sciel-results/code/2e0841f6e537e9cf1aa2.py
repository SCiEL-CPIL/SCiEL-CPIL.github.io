"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute neighbor lists for faster access
    adj_list = [sorted(adj[i]) for i in range(n)]
    
    deadline = time.time() + 80.0  # 80s budget, well under 90s
    best_path = []
    
    # Try each vertex as a starting point
    # Order starts by degree (lower degree first tends to be better for path extension)
    start_order = sorted(range(n), key=lambda v: (len(adj_list[v]), v))
    
    for start in start_order:
        if time.time() > deadline:
            break
        
        # DFS with Warnsdorff heuristic and backtracking
        path = [start]
        visited = [False] * n
        visited[start] = True
        
        def dfs(current):
            nonlocal best_path, deadline
            if time.time() > deadline:
                return False
            
            if len(path) > len(best_path):
                best_path = path[:]
            
            # Get unvisited neighbors, sorted by Warnsdorff rule
            neighbors = [v for v in adj_list[current] if not visited[v]]
            if not neighbors:
                return True  # dead end, backtrack
            
            # Warnsdorff: prioritize neighbors with fewer unvisited options
            neighbors.sort(key=lambda v: (sum(1 for w in adj_list[v] if not visited[w]), v))
            
            for v in neighbors:
                if time.time() > deadline:
                    return False
                visited[v] = True
                path.append(v)
                dfs(v)
                path.pop()
                visited[v] = False
                if time.time() > deadline:
                    return False
            return True
        
        dfs(start)
        if time.time() > deadline:
            break
    
    # Fallback: if somehow best_path is empty (shouldn't happen), return a single vertex
    if not best_path:
        best_path = [0]
    
    return best_path
# EVOLVE-BLOCK-END
