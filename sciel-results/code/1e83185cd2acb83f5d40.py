"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
START_TIME = time.time()
TIME_LIMIT = 85.0 # Safety margin

def construct():
    # Precompute arrival indices
    # arrival[u][v] = index (1-based)
    # Using a 2D list for fast access
    arrival = [[0] * N for _ in range(N)]
    for idx, edge in enumerate(P['edge_order']):
        u, v = edge
        arrival[u][v] = idx + 1
        arrival[v][u] = idx + 1
    
    # Helper to build adjacency masks for a given k
    def get_adj_masks(k):
        masks = [0] * N
        for u in range(N):
            mask = 0
            for v in range(N):
                if u != v and arrival[u][v] <= k:
                    mask |= (1 << v)
            masks[u] = mask
        return masks

    # DFS to find Hamilton cycle
    # Returns path if found, None otherwise
    def find_cycle(k):
        if time.time() - START_TIME > TIME_LIMIT:
            return None
            
        adj_masks = get_adj_masks(k)
        
        # Check if graph is connected enough? 
        # Simple check: each vertex must have degree >= 2
        for u in range(N):
            if adj_masks[u].bit_count() < 2:
                return None
        
        # Start from 0
        start = 0
        path = [start]
        visited_mask = 1 << start
        
        # Recursive DFS
        # Returns the path if a cycle is found, else None
        def dfs(u, visited_mask, path):
            if time.time() - START_TIME > TIME_LIMIT:
                return None
                
            if len(path) == N:
                # Check if u connects back to start
                if (adj_masks[u] >> start) & 1:
                    return path
                return None
            
            # Get candidates
            candidates = adj_masks[u] & ~visited_mask
            
            # Warnsdorff: sort by number of unvisited neighbors
            # We want to visit vertices with fewest options first
            # To avoid sorting overhead, we can just iterate and pick min, or sort
            # N=24, candidates <= 23. Sorting is fine.
            
            # Extract candidate vertices
            cand_vertices = []
            temp_mask = candidates
            while temp_mask:
                # Get lowest set bit
                lsb = temp_mask & -temp_mask
                v = lsb.bit_length() - 1
                cand_vertices.append(v)
                temp_mask ^= lsb
            
            # Sort by degree of unvisited neighbors
            # degree(v) = (adj_masks[v] & ~visited_mask).bit_count()
            # Note: visited_mask doesn't include v yet, so this is correct
            cand_vertices.sort(key=lambda v: (adj_masks[v] & ~visited_mask).bit_count())
            
            for v in cand_vertices:
                new_visited = visited_mask | (1 << v)
                new_path = path + [v]
                res = dfs(v, new_visited, new_path)
                if res is not None:
                    return res
            
            return None
        
        res = dfs(start, visited_mask, path)
        if res:
            return res
        return None

    # Binary search for minimal k
    lo = N # Min edges for a cycle is N
    hi = N * (N - 1) // 2 # Max edges
    best_k = hi
    best_path = None
    
    # First, check if a cycle exists at all with all edges
    # It should, as it's a complete graph.
    # But we want the minimal k.
    
    # Optimization: We can check connectivity or simple necessary conditions before full DFS
    # But DFS with Warnsdorff is usually fast enough for N=24.
    
    # To speed up binary search, we can do a linear scan if N is small, but binary is safer.
    # Actually, let's just do binary search.
    
    # We need to be careful: if find_cycle returns None, it means no cycle for that k.
    # If it returns a path, we update best.
    
    # Since we want the *minimal* k, we binary search for the first k where find_cycle(k) is not None.
    
    # However, find_cycle modifies path? No, it creates new lists.
    # But it's expensive.
    # Let's try to be smart.
    
    # If we find a cycle at k, we store it.
    # We continue searching lower k.
    
    # Standard binary search for first True
    while lo <= hi:
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        mid = (lo + hi) // 2
        path = find_cycle(mid)
        if path:
            best_path = path
            best_k = mid
            hi = mid - 1
        else:
            lo = mid + 1
            
    if best_path:
        return best_path
    else:
        # Fallback: return identity if something went wrong (should not happen)
        return list(range(N))

# EVOLVE-BLOCK-END
