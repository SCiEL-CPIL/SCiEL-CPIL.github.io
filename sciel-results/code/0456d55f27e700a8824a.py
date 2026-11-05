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
    
    def has_hamilton_cycle(K):
        """Check if first K edges contain a Hamilton cycle using optimized DFS."""
        # Build adjacency bitmasks
        adj_mask = [0] * n
        for i in range(K):
            u, v = edges[i]
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        
        # Quick check: all vertices must have degree >= 2
        for v in range(n):
            if bin(adj_mask[v]).count('1') < 2:
                return None
                
        # Quick connectivity check
        visited_mask = 1
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

        # DFS with bitmasks and forced moves
        # State: (current_vertex, visited_mask)
        # We fix the start vertex to 0 to break rotational symmetry.
        # The cycle must return to 0.
        
        # Precompute unvisited neighbors count for speed? 
        # It's dynamic, so we calculate on the fly.
        
        def dfs(u, visited):
            if time.time() > deadline:
                return None
            if visited == (1 << n) - 1:
                # Check if we can close the cycle
                if adj_mask[u] & (1 << 0):
                    return [0] # Placeholder, we need the path
                return None
            
            # Identify forced moves
            # A vertex v is forced if it has exactly 1 unvisited neighbor
            # Note: The start vertex 0 is special. If we are at the last step, 
            # the only unvisited vertex must be connected to 0.
            
            # Let's collect unvisited vertices
            unvisited_mask = ((1 << n) - 1) ^ visited
            
            # Check for forced moves among unvisited vertices
            # If an unvisited vertex v has only 1 neighbor in (visited | {u})? 
            # Actually, standard forced move: 
            # If an unvisited vertex v has degree 1 in the remaining graph 
            # (considering only unvisited vertices + current u), it must be visited next.
            
            # More precisely: 
            # For each unvisited vertex v, count neighbors in unvisited_mask | (1<<u).
            # If count is 1, that neighbor is forced.
            # If count is 0, dead end (unless v is the last one and connected to 0? No, if count 0 and v!=0, dead end).
            
            forced_next = -1
            has_forced = False
            
            # Iterate through unvisited vertices
            temp = unvisited_mask
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                temp ^= lsb
                
                # Count neighbors of v in (unvisited_mask | (1<<u))
                # This is the "remaining" graph including the current node u
                rem_mask = unvisited_mask | (1 << u)
                deg = bin(adj_mask[v] & rem_mask).count('1')
                
                if deg == 0:
                    return None # Dead end
                
                if deg == 1:
                    # The single neighbor is forced
                    # Find that neighbor
                    nb_mask = adj_mask[v] & rem_mask
                    # nb_mask should have exactly 1 bit
                    nb = (nb_mask & -nb_mask).bit_length() - 1
                    if not has_forced:
                        forced_next = nb
                        has_forced = True
                    else:
                        # Conflicting forced moves?
                        if forced_next != nb:
                            return None
            # End forced move check
            
            if has_forced:
                # Must visit forced_next
                if adj_mask[u] & (1 << forced_next):
                    return dfs(forced_next, visited | (1 << forced_next))
                else:
                    return None # Forced move not adjacent to current? Should not happen if logic is correct
            
            # No forced moves, use Warnsdorff heuristic
            neighbors = []
            mask = adj_mask[u] & unvisited_mask
            while mask:
                lsb = mask & (-mask)
                v = lsb.bit_length() - 1
                neighbors.append(v)
                mask ^= lsb
            
            if not neighbors:
                return None
                
            # Sort by number of unvisited neighbors (Warnsdorff)
            # Lower degree first
            def get_deg(v):
                return bin(adj_mask[v] & unvisited_mask).count('1')
            
            neighbors.sort(key=get_deg)
            
            for v in neighbors:
                res = dfs(v, visited | (1 << v))
                if res is not None:
                    return res
                    
            return None
        
        # We need to reconstruct the path. 
        # The above DFS returns None or a dummy. 
        # Let's modify DFS to return the path or use a global variable.
        
        path = []
        
        def dfs_path(u, visited):
            if time.time() > deadline:
                return False
            path.append(u)
            if visited == (1 << n) - 1:
                if adj_mask[u] & (1 << 0):
                    return True
                path.pop()
                return False
            
            unvisited_mask = ((1 << n) - 1) ^ visited
            
            forced_next = -1
            has_forced = False
            
            temp = unvisited_mask
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                temp ^= lsb
                
                rem_mask = unvisited_mask | (1 << u)
                deg = bin(adj_mask[v] & rem_mask).count('1')
                
                if deg == 0:
                    path.pop()
                    return False
                
                if deg == 1:
                    nb_mask = adj_mask[v] & rem_mask
                    nb = (nb_mask & -nb_mask).bit_length() - 1
                    if not has_forced:
                        forced_next = nb
                        has_forced = True
                    else:
                        if forced_next != nb:
                            path.pop()
                            return False
            
            if has_forced:
                if adj_mask[u] & (1 << forced_next):
                    if dfs_path(forced_next, visited | (1 << forced_next)):
                        return True
                path.pop()
                return False
            
            neighbors = []
            mask = adj_mask[u] & unvisited_mask
            while mask:
                lsb = mask & (-mask)
                v = lsb.bit_length() - 1
                neighbors.append(v)
                mask ^= lsb
            
            if not neighbors:
                path.pop()
                return False
                
            def get_deg(v):
                return bin(adj_mask[v] & unvisited_mask).count('1')
            
            neighbors.sort(key=get_deg)
            
            for v in neighbors:
                if dfs_path(v, visited | (1 << v)):
                    return True
                    
            path.pop()
            return False
        
        # Start DFS from 0
        # Note: 0 is already in path via dfs_path start?
        # dfs_path(0, 1) will append 0.
        if dfs_path(0, 1):
            return path
        return None

    # Binary search on K
    lo = n
    hi = len(edges)
    
    # First, verify hi works
    # It should, but let's be safe.
    
    while lo <= hi:
        if time.time() > deadline:
            break
        
        mid = (lo + hi) // 2
        
        cycle = has_hamilton_cycle(mid)
        
        if cycle is not None:
            best_cycle = cycle
            best_K = mid
            hi = mid - 1
        else:
            lo = mid + 1
            
    if best_cycle is None:
        # Fallback
        best_cycle = list(range(n))
        
    return best_cycle

if __name__ == "__main__":
    print(construct())
