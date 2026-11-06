import time
import random
import itertools

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    def popcount(x):
        return bin(x).count('1')
    
    def find_violation(adj, n):
        """
        Find a subset of size >= 6 that induces a regular subgraph.
        Returns (subset_mask, size) or (None, 0) if none found.
        """
        # Check sizes from 6 to n
        # For efficiency, we check smaller sizes first as they are more frequent violations
        for k in range(6, n + 1):
            # Iterate over all combinations of size k
            # Using itertools.combinations is slower than bitmask iteration for large n,
            # but for n=21, combinations of size 6 are 54k.
            # Bitmask iteration over 2^21 is 2M.
            # Let's stick to bitmask iteration for simplicity and speed of C-level operations?
            # Actually, iterating combinations in Python is slow.
            # Bitmask iteration:
            # for mask in range(1 << n):
            #   if popcount(mask) == k: ...
            # This is 2^n iterations.
            
            # Better: Generate masks of size k using Gosper's hack or similar?
            # Or just use itertools.combinations for k=6..10, and bitmask for larger?
            # For n=20, 2^20 = 1M. This is acceptable.
            
            # Let's just iterate all masks and filter by popcount.
            # This is simple and fast enough for n<=21.
            pass
        
        # Optimized approach:
        # Iterate masks from 0 to (1<<n)-1
        # Check popcount >= 6
        # Check regularity
        
        limit = 1 << n
        for mask in range(limit):
            k = popcount(mask)
            if k < 6:
                continue
            
            # Check regularity
            first_deg = -1
            is_regular = True
            count = 0
            
            # Iterate through vertices in mask
            # We can do this by iterating u from 0 to n-1
            for u in range(n):
                if (mask >> u) & 1:
                    deg = popcount(adj[u] & mask)
                    if count == 0:
                        first_deg = deg
                    else:
                        if deg != first_deg:
                            is_regular = False
                            break
                    count += 1
            
            if is_regular and count == k:
                return mask, k
        
        return None, 0

    def get_edges_from_adj(adj, n):
        edges = []
        for i in range(n):
            mask = adj[i]
            for j in range(i + 1, n):
                if (mask >> j) & 1:
                    edges.append([i, j])
        return edges

    def solve(n, time_budget):
        """
        Attempt to find a valid graph on n vertices within time_budget.
        """
        if time.time() - start_time > time_limit - time_budget:
            return None, None

        # Initialize random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        best_adj = adj[:]
        best_status = None
        
        # Local search
        while time.time() - start_time < time_limit - time_budget:
            mask, k = find_violation(adj, n)
            
            if mask is None:
                return n, get_edges_from_adj(adj, n)
            
            # Found violation. Flip a random edge within the violating subset.
            verts = [i for i in range(n) if (mask >> i) & 1]
            
            if len(verts) < 2:
                break # Should not happen for k>=6
                
            u, v = random.sample(verts, 2)
            
            # Flip edge (u, v)
            if (adj[u] >> v) & 1:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            # Occasionally restart if stuck?
            # Simple restart strategy: if no improvement in X steps, restart?
            # For now, just keep flipping.
            
        return None, None

    # Try n=21 first
    # Give it 100 seconds
    result = solve(21, 100.0)
    if result[0] is not None:
        return {'n': result[0], 'edges': result[1]}
        
    # Try n=20
    # Give it 40 seconds
    result = solve(20, 40.0)
    if result[0] is not None:
        return {'n': result[0], 'edges': result[1]}
        
    # Fallback: Return a trivially valid graph (n=1) to avoid score 0
    # Or a known small valid graph?
    # n=5 is always valid?
    # Any graph on 5 vertices has no subset of size 6.
    # So any graph on n<6 is valid.
    # Let's return a random graph on n=5.
    adj = [0] * 5
    for i in range(5):
        for j in range(i + 1, 5):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return {'n': 5, 'edges': get_edges_from_adj(adj, 5)}
