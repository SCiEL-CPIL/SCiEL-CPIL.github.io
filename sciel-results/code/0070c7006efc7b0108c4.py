import time
import sys
from collections import defaultdict

# Increase recursion limit for safety, though 21 is small
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
    deadline = start_time + 55.0  # 5s margin
    
    # Normalize edges to (min, max) tuples for consistent hashing
    normalized_edges = set()
    for e in P['edges']:
        u, v = e
        if u > v:
            u, v = v, u
        normalized_edges.add((u, v))
        
    # Build adjacency list using normalized neighbors
    adj = defaultdict(list)
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
        
    # Sort neighbors to have a deterministic order for backtracking
    for node in adj:
        adj[node].sort()
        
    remaining_edges = set(normalized_edges)
    
    def get_edge_key(u, v):
        if u > v:
            u, v = v, u
        return (u, v)
        
    def dfs(path, used_vertices, current):
        if len(path) == n:
            # Check if we can close the cycle
            edge_key = get_edge_key(current, path[0])
            if edge_key in remaining_edges:
                return path[:]
            return None
            
        # Heuristic: Try neighbors with fewer remaining options first (Warnsdorff)
        # But we must only consider edges in remaining_edges
        candidates = []
        for neighbor in adj[current]:
            if neighbor in used_vertices:
                continue
            edge_key = get_edge_key(current, neighbor)
            if edge_key in remaining_edges:
                # Count remaining edges from neighbor to unvisited vertices
                # This is expensive, so maybe just sort by degree in remaining graph?
                # For small n, simple order might suffice if randomized or varied.
                # Let's stick to a deterministic order but try reversing for diversity if needed?
                # Actually, for this specific graph, the structure is very regular.
                candidates.append(neighbor)
                
        # Sort candidates: prefer neighbors that have fewer available exits (Warnsdorff-like)
        def heuristic(nb):
            count = 0
            for nn in adj[nb]:
                if nn in used_vertices:
                    continue
                if get_edge_key(nb, nn) in remaining_edges:
                    count += 1
            return count
            
        candidates.sort(key=heuristic)
        
        for neighbor in candidates:
            edge_key = get_edge_key(current, neighbor)
            remaining_edges.remove(edge_key)
            used_vertices.add(neighbor)
            path.append(neighbor)
            
            result = dfs(path, used_vertices, neighbor)
            if result is not None:
                return result
                
            # Backtrack
            path.pop()
            used_vertices.remove(neighbor)
            remaining_edges.add(edge_key)
            
        return None
        
    cycles = []
    
    # We expect at most 3 cycles
    for _ in range(3):
        if time.time() > deadline:
            break
            
        # Start from vertex 0 to break symmetry
        start_vertex = 0
        initial_path = [start_vertex]
        initial_visited = {start_vertex}
        
        found_cycle = dfs(initial_path, initial_visited, start_vertex)
        
        if found_cycle is None:
            break
            
        cycles.append(found_cycle)
        
    # Fallback if no cycles found (should not happen)
    if not cycles:
        # Try the identity cycle
        ident_cycle = list(range(n))
        # Verify it's valid
        valid = True
        for i in range(n):
            u = ident_cycle[i]
            v = ident_cycle[(i + 1) % n]
            if get_edge_key(u, v) not in P['edges']:
                # P['edges'] is list of lists, so we need to check normalized
                # Let's just check against the set
                if get_edge_key(u, v) not in normalized_edges:
                    valid = False
                    break
        if valid:
            cycles.append(ident_cycle)
            
    return cycles
