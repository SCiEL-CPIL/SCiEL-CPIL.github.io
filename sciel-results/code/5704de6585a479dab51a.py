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
    
    def popcount(x):
        return x.bit_count() if hasattr(int, 'bit_count') else bin(x).count('1')
    
    def has_hamilton_cycle(K, time_limit):
        """Check if first K edges contain a Hamilton cycle using optimized search."""
        # Build adjacency bitmasks and degree counts
        adj_mask = [0] * n
        degree = [0] * n
        for i in range(K):
            u, v = edges[i]
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            degree[u] += 1
            degree[v] += 1
        
        # Quick checks
        # 1. Minimum degree >= 2
        for v in range(n):
            if degree[v] < 2:
                return None
        
        # 2. Connectivity check using BFS with bitmasks
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
        
        # 3. Check 2-connectivity (no articulation points) using iterative Tarjan
        disc = [-1] * n
        low = [-1] * n
        parent = [-1] * n
        ap = [False] * n
        time_counter = [0]
        
        # Iterative Tarjan's algorithm
        stack = [(0, -1, 0)]  # (node, parent, next_child_index)
        disc[0] = low[0] = 0
        time_counter[0] = 1
        child_count = [0] * n
        
        while stack:
            u, par, idx = stack[-1]
            mask = adj_mask[u]
            # Find the next unvisited neighbor
            found_next = False
            while idx < n:
                if (mask >> idx) & 1:
                    v = idx
                    if disc[v] == -1:
                        child_count[u] += 1
                        parent[v] = u
                        disc[v] = low[v] = time_counter[0]
                        time_counter[0] += 1
                        stack.append((v, u, 0))
                        found_next = True
                        break
                    elif v != par:
                        low[u] = min(low[u], disc[v])
                idx += 1
            
            if found_next:
                stack[-1] = (u, par, idx)
            else:
                # All neighbors processed, pop
                stack.pop()
                if stack:
                    pu, _, _ = stack[-1]
                    low[pu] = min(low[pu], low[u])
                    if pu == 0 and child_count[pu] > 1:
                        ap[pu] = True
                    elif pu != 0 and low[u] >= disc[pu]:
                        ap[pu] = True
        
        if any(ap):
            return None
        
        # Now do DFS to find Hamilton cycle with multiple strategies
        full_mask = (1 << n) - 1
        check_counter = [0]
        
        def find_cycle(start_vertex, time_limit):
            # Precompute neighbor lists
            neigh = [[] for _ in range(n)]
            for v in range(n):
                mask = adj_mask[v]
                while mask:
                    lsb = mask & (-mask)
                    u = lsb.bit_length() - 1
                    neigh[v].append(u)
                    mask ^= lsb
            
            # Try multiple random orderings with different strategies
            for attempt in range(40):
                if time.time() > time_limit:
                    return None
                
                rng_local = random.Random(attempt * 7919 + 13 + start_vertex * 31)
                
                visited_mask = 1 << start_vertex
                path = [start_vertex]
                
                # Strategy 1: Warnsdorff with random tie-breaking
                # Strategy 2: Random order
                # Strategy 3: Prefer high-degree unvisited neighbors
                
                strategy = attempt % 3
                
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
                    
                    if strategy == 0:
                        # Warnsdorff: fewest unvisited neighbors first
                        neighbors.sort(key=lambda v: popcount(adj_mask[v] & ~vmask))
                    elif strategy == 1:
                        # Random order
                        rng_local.shuffle(neighbors)
                    else:
                        # Prefer high total degree
                        neighbors.sort(key=lambda v: -degree[v])
                    
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
        
        # Try starting from vertex 0 first, then others
        for sv in range(n):
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
    
    if best_cycle is None:
        cycle = has_hamilton_cycle(len(edges), deadline - 1.0)
        if cycle is not None:
            best_cycle = cycle
        else:
            best_cycle = list(range(n))
    
    return best_cycle

if __name__ == "__main__":
    print(construct())
