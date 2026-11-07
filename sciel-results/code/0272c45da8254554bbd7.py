import time
import sys
from collections import defaultdict

# Increase recursion limit for safety
sys.setrecursionlimit(10000)

P = {
    'n': 21, 
    'edges': [
        [0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], 
        [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], 
        [2, 3], [2, 4], [2, 7], [2, 18], 
        [3, 4], [3, 5], [3, 8], [3, 19], 
        [4, 5], [4, 6], [4, 9], [4, 20], 
        [5, 6], [5, 7], [5, 10], 
        [6, 7], [6, 8], [6, 11], 
        [7, 8], [7, 9], [7, 12], 
        [8, 9], [8, 10], [8, 13], 
        [9, 10], [9, 11], [9, 14], 
        [10, 11], [10, 12], [10, 15], 
        [11, 12], [11, 13], [11, 16], 
        [12, 13], [12, 14], [12, 17], 
        [13, 14], [13, 15], [13, 18], 
        [14, 15], [14, 16], [14, 19], 
        [15, 16], [15, 17], [15, 20], 
        [16, 17], [16, 18], 
        [17, 18], [17, 19], 
        [18, 19], [18, 20], 
        [19, 20]
    ]
}

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 50.0  # 10s margin before 60s limit
    
    # Map edges to indices for bitmask representation
    edge_map = {}
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(P['edges']):
        if u > v:
            u, v = v, u
        edge_map[(u, v)] = i
        adj[u].append((v, i))
        adj[v].append((u, i))
        
    total_edges = len(P['edges'])
    
    def get_edge_index(u, v):
        if u > v:
            u, v = v, u
        return edge_map.get((u, v), -1)

    def find_cycle(used_edges_mask):
        # Start from vertex 0
        start = 0
        path = [start]
        visited_mask = 1 << start
        
        def dfs(current, path, visited_mask, used_edges_mask):
            if len(path) == n:
                # Check closing edge
                e_idx = get_edge_index(current, path[0])
                if e_idx != -1 and not (used_edges_mask & (1 << e_idx)):
                    return path[:]
                return None
            
            # Heuristic: try neighbors with fewest available edges first (Warnsdorff)
            candidates = []
            for neighbor, e_idx in adj[current]:
                if visited_mask & (1 << neighbor):
                    continue
                if used_edges_mask & (1 << e_idx):
                    continue
                # Count available edges for neighbor
                avail = 0
                for nn, ee in adj[neighbor]:
                    if not (visited_mask & (1 << nn)) and not (used_edges_mask & (1 << ee)):
                        avail += 1
                candidates.append((avail, neighbor, e_idx))
                
            if not candidates:
                return None

            candidates.sort(key=lambda x: x[0])
            
            for avail, neighbor, e_idx in candidates:
                new_used_mask = used_edges_mask | (1 << e_idx)
                new_visited_mask = visited_mask | (1 << neighbor)
                path.append(neighbor)
                
                res = dfs(neighbor, path, new_visited_mask, new_used_mask)
                if res is not None:
                    return res
                
                path.pop()
            
            return None

        return dfs(start, path, visited_mask, used_edges_mask)

    cycles = []
    used_edges_mask = 0
    
    # Try to find 3 cycles
    for _ in range(3):
        if time.time() > deadline:
            break
            
        cycle = find_cycle(used_edges_mask)
        if cycle is None:
            # If deterministic fails, try a few random restarts?
            # For now, break.
            break
        
        cycles.append(cycle)
        
        # Update used edges mask
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            e_idx = get_edge_index(u, v)
            if e_idx != -1:
                used_edges_mask |= (1 << e_idx)
            else:
                # This should not happen if cycle is valid
                break
            
    # If we found fewer than 3, we can't do better.
    # Return the list of cycles found.
    
    return cycles
