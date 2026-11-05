def construct():
    import time
    import random

    start_time = time.time()
    DEADLINE = 115.0
    n = 40
    random.seed(12345)

    best_edges = None
    best_count = 0

    def bitset_to_edges(adj):
        return [[u, v] for u in range(n) for v in range(u + 1, n) if (adj[u] >> v) & 1]

    def count_edges(adj):
        return sum(bin(x).count('1') for x in adj) // 2

    def can_add_edge(adj, u, v):
        """
        Check if adding edge (u, v) creates a C6.
        A C6 through (u, v) is a path u-a-b-c-d-v where a,b,c,d are distinct and not u,v.
        """
        # S1: potential 'a' nodes (neighbors of u, excluding v)
        S1 = adj[u] & ~(1 << u) & ~(1 << v)
        
        if S1 == 0:
            return True # No path of length 1 from u, so no C6

        # S2: potential 'b' nodes (neighbors of S1, excluding u, v)
        S2 = 0
        a = S1
        while a:
            lsb = a & -a
            a ^= lsb
            idx = lsb.bit_length() - 1
            S2 |= adj[idx]
        S2 &= ~(1 << u) & ~(1 << v)
        
        if S2 == 0:
            return True # No path of length 2

        # S3: potential 'c' nodes (neighbors of S2, excluding u, v, S1, S2)
        # c must be distinct from a (in S1) and b (in S2)
        S3 = 0
        b = S2
        while b:
            lsb = b & -b
            b ^= lsb
            idx = lsb.bit_length() - 1
            S3 |= adj[idx]
        S3 &= ~(1 << u) & ~(1 << v)
        S3 &= ~S1
        S3 &= ~S2
        
        if S3 == 0:
            return True # No path of length 3

        # S4: potential 'd' nodes (neighbors of S3, excluding u, v, S1, S2, S3)
        # d must be distinct from a, b, c
        S4 = 0
        c = S3
        while c:
            lsb = c & -c
            c ^= lsb
            idx = lsb.bit_length() - 1
            S4 |= adj[idx]
        S4 &= ~(1 << u) & ~(1 << v)
        S4 &= ~S1
        S4 &= ~S2
        S4 &= ~S3
        
        # Check if any d in S4 is connected to v
        if (S4 & adj[v]) != 0:
            return False # C6 found
        
        return True

    def add_all_possible(adj, deadline):
        """Iteratively add all edges that don't create a C6."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Randomize edge order to escape symmetries
            candidates = [(u, v) for u in range(n) for v in range(u + 1, n) if not ((adj[u] >> v) & 1)]
            random.shuffle(candidates)
            for u, v in candidates:
                if time.time() > deadline:
                    return
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    changed = True

    def local_search_swap(adj, deadline):
        """
        Try to improve by removing one edge and adding others.
        """
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Get list of current edges
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj[u] >> v) & 1:
                        edges.append((u, v))
            
            if not edges:
                break
            
            # Try a few random swaps
            for _ in range(10):
                if time.time() > deadline:
                    break
                u, v = random.choice(edges)
                
                # Save state
                old_adj = adj[:]
                old_count = count_edges(old_adj)
                
                # Remove edge
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                
                # Try to add all possible edges
                add_all_possible(adj, min(deadline, time.time() + 0.1))
                
                new_count = count_edges(adj)
                
                if new_count > old_count:
                    improved = True
                    # Keep the change
                else:
                    # Revert
                    adj = old_adj
                    break # Restart loop with new random choice? Or just break inner?
                    # Let's break inner and try another edge
                    # Actually, if we revert, we should try another edge.
                    # But 'improved' flag is for outer loop.
                    # Let's just continue to next swap attempt.
                    # Wait, if we revert, adj is restored.
                    # We should break the inner loop to pick a new random edge?
                    # Or just continue.
                    pass

    # Main search loop
    restarts = 0
    while time.time() < DEADLINE - 5:
        restarts += 1
        random.seed(12345 + restarts)
        
        # Initialize with 8 disjoint K5
        adj = [0] * n
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    u = 5 * k + a
                    v = 5 * k + b
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Greedy fill
        add_all_possible(adj, min(time.time() + 2.0, DEADLINE - 5))
        
        # Local search
        local_search_swap(adj, min(time.time() + 3.0, DEADLINE - 5))
        
        count = count_edges(adj)
        if count > best_count:
            best_count = count
            best_edges = bitset_to_edges(adj)
            
    if best_edges is None:
        # Fallback: 8 K5
        best_edges = []
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    best_edges.append([5 * k + a, 5 * k + b])

    return best_edges
