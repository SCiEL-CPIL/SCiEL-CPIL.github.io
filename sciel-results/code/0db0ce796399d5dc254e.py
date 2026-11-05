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
    
    # Precompute popcount
    try:
        (1).bit_count()
        def popcount(x):
            return x.bit_count()
    except AttributeError:
        def popcount(x):
            return bin(x).count('1')
            
    full_mask = (1 << n) - 1

    def is_connected(adj, mask):
        """Check if subgraph induced by vertices in 'mask' is connected."""
        if mask == 0:
            return True
        # Find first vertex
        v0_lsb = mask & -mask
        v0 = v0_lsb.bit_length() - 1
        
        visited = 0
        frontier = 1 << v0
        
        while frontier:
            visited |= frontier
            new_frontier = 0
            temp = frontier
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                # Get neighbors of v within mask
                new_frontier |= adj[v] & mask
                temp ^= lsb
            new_frontier &= ~visited
            frontier = new_frontier
            
        return (visited == mask)

    def has_hamilton_cycle(K, time_limit):
        """Check if first K edges contain a Hamilton cycle."""
        # Build adjacency bitmasks
        adj = [0] * n
        for i in range(K):
            u, v = edges[i]
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Quick connectivity check on whole graph
        if not is_connected(adj, full_mask):
            return None
            
        # Check minimum degree
        for v in range(n):
            if popcount(adj[v]) < 2:
                return None
                
        # Iterative DFS with Warnsdorff and Connectivity Pruning
        # We try a few random permutations of tie-breaks
        
        def dfs(u, visited, path, attempt_seed):
            # Time check
            if time.time() > time_limit:
                return None
                
            if len(path) == n:
                if adj[u] & (1 << path[0]):
                    return path[:]
                return None
            
            unvisited = full_mask ^ visited
            
            # Pruning: The unvisited vertices must be connected
            # Note: We are at u. We will move to a neighbor v in unvisited.
            # The path will continue from v through all other unvisited vertices.
            # So the subgraph induced by unvisited must be connected.
            # Actually, if we remove u, the unvisited vertices must be connected?
            # Yes, because we enter at v and must exit to start (or just finish).
            # The sequence of unvisited vertices must form a path.
            # Thus, they must be connected in the induced subgraph.
            if not is_connected(adj, unvisited):
                return None
                
            # Collect neighbors
            neighbors = []
            mask = adj[u] & unvisited
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                # Warnsdorff: count unvisited neighbors of v
                # We exclude 'u' from unvisited already, so just count in unvisited
                deg = popcount(adj[v] & unvisited)
                neighbors.append((deg, v))
                mask ^= lsb
                
            if not neighbors:
                return None
                
            # Sort by degree (ascending)
            neighbors.sort(key=lambda x: x[0])
            
            # To add randomness, we can shuffle equal-degree neighbors
            # But simple sort is often enough. Let's add a small randomization for ties.
            # Group by degree
            deg_groups = {}
            for deg, v in neighbors:
                if deg not in deg_groups:
                    deg_groups[deg] = []
                deg_groups[deg].append(v)
                
            ordered_neighbors = []
            for deg in sorted(deg_groups.keys()):
                group = deg_groups[deg]
                # Shuffle group with deterministic seed based on attempt
                random.Random(attempt_seed * 1000 + deg).shuffle(group)
                ordered_neighbors.extend(group)
                
            for v in ordered_neighbors:
                path.append(v)
                result = dfs(v, visited | (1 << v), path, attempt_seed)
                path.pop()
                if result is not None:
                    return result
                    
            return None
            
        # Try starting from vertex 0
        # Try a few attempts with different random seeds for tie-breaking
        for attempt in range(20):
            if time.time() > time_limit:
                break
            cycle = dfs(0, 1 << 0, [0], attempt)
            if cycle is not None:
                return cycle
                
        # If failed, try starting from other vertices? 
        # Due to symmetry, 0 is usually fine, but graph might be asymmetric.
        for sv in [1, 2]:
            if time.time() > time_limit:
                break
            for attempt in range(5):
                cycle = dfs(sv, 1 << sv, [sv], attempt + 100)
                if cycle is not None:
                    return cycle
                    
        return None

    # Binary search on K
    lo = n
    hi = len(edges)
    
    found_any = False
    while lo <= hi:
        if time.time() > deadline:
            break
            
        mid = (lo + hi) // 2
        
        # Use a smaller time limit for each check to allow multiple checks
        # But we need to be careful. If we spend too much on one, we fail.
        # Let's allocate time dynamically.
        # If we have 80s left, and 8 steps left, ~10s per step.
        time_left = deadline - time.time()
        if time_left < 0.1:
            break
            
        # Heuristic: give more time to smaller K? No, smaller K is harder to find?
        # Actually, smaller K is sparser, DFS might prune faster.
        # Larger K is denser, DFS might get stuck in loops?
        # Let's just use a fixed fraction or a safe limit.
        check_limit = min(time_left - 1.0, 5.0) # Max 5s per check, min 1s safety
        
        cycle = has_hamilton_cycle(mid, check_limit)
        
        if cycle is not None:
            best_cycle = cycle
            best_K = mid
            found_any = True
            hi = mid - 1
        else:
            lo = mid + 1
            
    if best_cycle is None:
        # Fallback: try the full graph
        cycle = has_hamilton_cycle(len(edges), deadline - 1.0)
        if cycle is not None:
            best_cycle = cycle
        else:
            # Absolute fallback: identity permutation (might be invalid if edges missing? No, complete graph)
            # But identity permutation edges might have high indices.
            # Just return a valid cycle from the full graph.
            # Since K=276 is complete, a cycle always exists.
            # The DFS should have found it. If not, something is wrong.
            # Let's try a simple greedy construction for the full graph as last resort.
            best_cycle = list(range(n))
            
    return best_cycle

if __name__ == "__main__":
    print(construct())
