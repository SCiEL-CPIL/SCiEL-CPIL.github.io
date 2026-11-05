import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 88.0  # 2s margin for the 90s limit
    
    n = 24
    rng = random.Random(614)
    
    # Generate the shuffled edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    rng.shuffle(edges)
    
    # Map edge to its 1-based position
    pos = {}
    for i, (u, v) in enumerate(edges):
        pos[(u, v)] = i + 1
        pos[(v, u)] = i + 1
        
    best_cycle = None
    best_K = float('inf')
    
    # Precompute adjacency as bitmasks for fast operations
    # adj_mask[v] has bit u set if edge (u,v) exists
    
    def has_hamilton_cycle(K, time_limit):
        """Check if first K edges contain a Hamilton cycle using bitmask DFS."""
        # Build adjacency bitmasks
        adj_mask = [0] * n
        for i in range(K):
            u, v = edges[i]
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        
        # Quick connectivity check using BFS with bitmasks
        visited_mask = 1  # vertex 0
        frontier = 1
        while frontier:
            # Get all neighbors of frontier vertices
            new_mask = 0
            temp = frontier
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                new_mask |= adj_mask[v]
                temp ^= lsb
            new_mask &= ~visited_mask
            if not new_mask:
                break
            visited_mask |= new_mask
            frontier = new_mask
        
        if visited_mask != (1 << n) - 1:
            return None
        
        # Bitmask DFS with Warnsdorff heuristic
        # State: (current_vertex, visited_mask, path)
        # Use iterative DFS to avoid recursion limits
        
        def dfs_bitmask(start_vertex):
            """Find Hamilton cycle starting and ending at start_vertex."""
            target_mask = (1 << n) - 1
            # Stack entries: (vertex, visited_mask, path_list)
            # We'll use a stack-based approach with backtracking
            
            # Try multiple starting points and orderings
            for attempt in range(20):
                if time.time() > time_limit:
                    return None
                
                # Shuffle the order of neighbor exploration
                rng_local = random.Random(attempt * 7919 + 13)
                
                visited = [False] * n
                path = [start_vertex]
                visited[start_vertex] = True
                visited_mask = 1 << start_vertex
                
                # Iterative DFS with explicit stack
                # Stack stores: (vertex, next_neighbor_idx, path_snapshot)
                # But path_snapshot is expensive. Instead, use recursive-like with undo.
                
                # Actually, let's use a different approach: 
                # Precompute for each vertex, the list of neighbors sorted by degree
                # Then do DFS with backtracking
                
                # For speed, use a non-recursive approach with explicit state
                # State: (current, visited_mask, path)
                # We'll use a stack of (vertex, visited_mask, path_tuple)
                # But tuples are expensive for 24 elements...
                
                # Let me try a simpler recursive approach but with better pruning
                
                def dfs_rec(u, vmask, pth):
                    if time.time() > time_limit:
                        return None
                    if len(pth) == n:
                        if (vmask >> u) & 1 and adj_mask[u] & (1 << start_vertex):
                            return pth[:]
                        return None
                    
                    # Get unvisited neighbors of u
                    neighbors = []
                    mask = adj_mask[u] & ~vmask
                    while mask:
                        lsb = mask & (-mask)
                        v = lsb.bit_length() - 1
                        neighbors.append(v)
                        mask ^= lsb
                    
                    if not neighbors:
                        return None
                    
                    # Warnsdorff: sort by number of unvisited neighbors
                    def get_unvisited_deg(v):
                        return bin(adj_mask[v] & ~vmask).count('1')
                    
                    # Shuffle then stable sort by degree
                    rng_local.shuffle(neighbors)
                    neighbors.sort(key=get_unvisited_deg)
                    
                    for v in neighbors:
                        new_pth = pth + [v]
                        result = dfs_rec(v, vmask | (1 << v), new_pth)
                        if result is not None:
                            return result
                    
                    return None
                
                result = dfs_rec(start_vertex, visited_mask, path)
                if result is not None:
                    return result
            
            return None
        
        # Try starting from vertex 0
        cycle = dfs_bitmask(0)
        if cycle is not None:
            return cycle
        
        return None
    
    # Binary search on K
    lo = n  # Minimum edges for a cycle is n
    hi = len(edges)
    
    # First, verify that hi works (it should, since K_{24} has Hamilton cycles)
    # But we need to be careful about time
    
    found_any = False
    while lo <= hi:
        if time.time() > deadline:
            break
        
        mid = (lo + hi) // 2
        
        # Try to find a Hamilton cycle in G_mid
        cycle = has_hamilton_cycle(mid, deadline - 2.0)
        
        if cycle is not None:
            best_cycle = cycle
            best_K = mid
            found_any = True
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_cycle is None:
        # Fallback: return a cycle that definitely exists
        # Try with all edges
        cycle = has_hamilton_cycle(len(edges), deadline - 1.0)
        if cycle is not None:
            best_cycle = cycle
        else:
            # Last resort: identity permutation
            best_cycle = list(range(n))
    
    return best_cycle

if __name__ == "__main__":
    print(construct())
