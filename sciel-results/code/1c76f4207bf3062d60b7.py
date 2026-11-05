import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin for the 90s limit
    
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
    
    # Precompute popcount
    try:
        (1).bit_count()
        def popcount(x):
            return x.bit_count()
    except AttributeError:
        def popcount(x):
            return bin(x).count('1')
    
    def has_hamilton_cycle(K, time_limit):
        """Check if first K edges contain a Hamilton cycle using optimized bitmask DFS."""
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
        
        # Check minimum degree: every vertex must have degree >= 2
        for v in range(n):
            if popcount(adj_mask[v]) < 2:
                return None
        
        # Iterative DFS with Warnsdorff heuristic
        full_mask = (1 << n) - 1
        
        def find_cycle(start_vertex, time_limit):
            # Try multiple random orderings
            for attempt in range(50):
                if time.time() > time_limit:
                    return None
                
                rng_local = random.Random(attempt * 7919 + 13)
                
                visited_mask = 1 << start_vertex
                path = [start_vertex]
                
                # Precompute neighbor lists for each vertex
                neigh = [[] for _ in range(n)]
                for v in range(n):
                    mask = adj_mask[v]
                    while mask:
                        lsb = mask & (-mask)
                        u = lsb.bit_length() - 1
                        neigh[v].append(u)
                        mask ^= lsb
                
                # Shuffle neighbor lists for diversity
                for v in range(n):
                    rng_local.shuffle(neigh[v])
                
                # Recursive DFS with better pruning
                check_counter = [0]
                
                def dfs(u, vmask, pth):
                    check_counter[0] += 1
                    if check_counter[0] % 500 == 0 and time.time() > time_limit:
                        return None
                    
                    if len(pth) == n:
                        if adj_mask[u] & (1 << start_vertex):
                            return pth[:]
                        return None
                    
                    # Get unvisited neighbors
                    mask = adj_mask[u] & ~vmask
                    if not mask:
                        return None
                    
                    # Collect neighbors
                    neighbors = []
                    m = mask
                    while m:
                        lsb = m & (-m)
                        v = lsb.bit_length() - 1
                        neighbors.append(v)
                        m ^= lsb
                    
                    # Warnsdorff: sort by number of unvisited neighbors (ascending)
                    neighbors.sort(key=lambda v: popcount(adj_mask[v] & ~vmask))
                    
                    for v in neighbors:
                        pth.append(v)
                        result = dfs(v, vmask | (1 << v), pth)
                        pth.pop()
                        if result is not None:
                            return result
                    
                    return None
                
                result = dfs(start_vertex, visited_mask, path)
                if result is not None:
                    return result
            
            return None
        
        # Try starting from vertex 0
        cycle = find_cycle(0, time_limit)
        if cycle is not None:
            return cycle
        
        # If that fails, try a few other start vertices
        for sv in [1, 2, 3]:
            if time.time() > time_limit:
                break
            cycle = find_cycle(sv, time_limit)
            if cycle is not None:
                return cycle
        
        return None
    
    # Binary search on K
    lo = n  # Minimum edges for a cycle is n
    hi = len(edges)
    
    found_any = False
    while lo <= hi:
        if time.time() > deadline:
            break
        
        mid = (lo + hi) // 2
        
        cycle = has_hamilton_cycle(mid, deadline - 2.0)
        
        if cycle is not None:
            best_cycle = cycle
            best_K = mid
            found_any = True
            hi = mid - 1
        else:
            lo = mid + 1
    
    # If we found a cycle, try to verify it's truly minimal by checking K-1
    if best_cycle is not None and best_K > n:
        if time.time() < deadline - 5.0:
            # Try to find a cycle with K-1 edges to see if we can go lower
            for check_K in range(best_K - 1, max(n, best_K - 10), -1):
                if time.time() > deadline - 2.0:
                    break
                cycle = has_hamilton_cycle(check_K, deadline - 2.0)
                if cycle is not None:
                    best_cycle = cycle
                    best_K = check_K
                else:
                    break
    
    if best_cycle is None:
        cycle = has_hamilton_cycle(len(edges), deadline - 1.0)
        if cycle is not None:
            best_cycle = cycle
        else:
            best_cycle = list(range(n))
    
    return best_cycle

if __name__ == "__main__":
    print(construct())
