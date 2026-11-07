import time

def construct():
    """
    Find m in [0, 1000000] that maximizes f(12, m).
    f(12, m) is the least L > 0 such that (m, m+L] contains 12 distinct integers
    b_1, ..., b_12 with i | b_i for each i in 1..12.
    
    The function has period 27720 (LCM of 1..12).
    We precompute f(12, m) for all m in [0, 27719] using a fast bitset-based
    bipartite matching algorithm, then return the m that maximizes f.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # 80s deadline to be safe
    
    n = 12
    period = 27720  # LCM(1, 2, ..., 12)
    
    # Precompute for each i in 1..12 and each residue r in 0..period-1,
    # the bitset of positions p in 0..period-1 such that (p+1) ≡ -m (mod i)
    # when m ≡ r (mod period).
    # Actually, for a fixed m, the valid positions for i are those p where
    # (m + p + 1) % i == 0, i.e., p ≡ (-m-1) % i (mod i).
    # Within a window of length L, we need to find if there's a matching.
    
    # Strategy: For each m in [0, period), compute the minimal L.
    # We use a sliding window approach with bitsets.
    
    # For efficiency, we'll use a direct approach:
    # For each m, try L = 1, 2, 3, ... until we find a matching.
    # Since L is small (<= 144), and we have 27720 values of m,
    # we need the matching check to be very fast.
    
    # Precompute: for each i in 1..12, and for each residue class mod i,
    # store a bitset (as an integer) of positions in [0, period) that match.
    
    # Actually, let's use a different approach:
    # For a given m and L, the candidate positions for i are:
    # p_0 = (-m-1) % i, then p_0, p_0+i, p_0+2i, ... within [0, L-1]
    # (where position p corresponds to number m+p+1)
    
    # Let's implement a fast matching using bitsets.
    # For n=12 and L<=144, we can use a simple DFS matching with bitsets.
    
    def can_match_fast(m, L):
        """Check if there's a valid matching for interval (m, m+L]."""
        # For each i in 1..12, compute the set of valid positions (0-indexed, 0..L-1)
        # as a bitmask.
        # Then use a greedy/DFS matching.
        
        # For i=1, all positions are valid.
        # For i>1, positions p where (m+p+1) % i == 0.
        
        # Build adjacency as bitmasks
        adj = [0] * n  # adj[i-1] is bitmask of valid positions for i
        
        for i in range(1, n + 1):
            mask = 0
            rem = (-m - 1) % i
            p = rem
            while p < L:
                mask |= (1 << p)
                p += i
            adj[i-1] = mask
        
        # Now do bipartite matching using DFS with bitmasks
        # match_r[p] = i (0-indexed) matched to position p, or -1
        match_r = [-1] * L
        
        def dfs(i, visited):
            mask = adj[i]
            # Iterate over set bits in mask
            m2 = mask
            while m2:
                # Get lowest set bit
                lb = m2 & (-m2)
                p = lb.bit_length() - 1
                m2 ^= lb
                if visited[p]:
                    continue
                visited[p] = True
                if match_r[p] == -1 or dfs(match_r[p], visited):
                    match_r[p] = i
                    return True
            return False
        
        count = 0
        for i in range(n):
            visited = [False] * L
            if dfs(i, visited):
                count += 1
        return count == n
    
    best_m = 0
    best_L = 0
    
    # Precompute f(m) for all m in [0, period)
    # Optimization: use the fact that f(m) depends only on m mod period.
    
    # We'll iterate m from 0 to period-1, and for each, find minimal L.
    # To speed up, we can use a lower bound: L must be at least max(i) = 12,
    # and also must be large enough to accommodate the "hardest" i.
    
    # Actually, let's just do a linear scan for L for each m.
    # Since L is small, this should be fast enough.
    
    # Further optimization: precompute the "difficulty" of each m.
    # The key constraint is that for each i, we need at least one multiple
    # of i in the window, and all b_i must be distinct.
    
    # Let's try a different approach: for each L, find all m that work.
    # This might be faster if we can vectorize.
    
    # Actually, let's just do the straightforward approach with optimizations.
    
    for m in range(period):
        if time.time() > deadline:
            break
        
        # Find minimal L for this m
        # Lower bound: L >= 12 (need at least 12 distinct numbers)
        # Upper bound: 144
        found = False
        for L in range(12, 145):
            if can_match_fast(m, L):
                if L > best_L:
                    best_L = L
                    best_m = m
                found = True
                break
        
        if not found:
            # Should not happen, but handle gracefully
            pass
    
    # The best_m is in [0, period-1], which is within [0, 1000000].
    # Return as Python int.
    return int(best_m)

if __name__ == "__main__":
    construct()
