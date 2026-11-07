import time
import math

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    n = 12
    period = 27720  # lcm(1..12)
    max_L = 144
    
    # Precompute bitmasks for valid offsets
    # For each i, for each residue rem (0..i-1), bitmask of d in 1..max_L where d % i == rem
    valid_masks = {}
    for i in range(1, n + 1):
        valid_masks[i] = [0] * i
        for d in range(1, max_L + 1):
            rem = d % i
            valid_masks[i][rem] |= (1 << d)
    
    # Precompute popcount for all possible mask values (up to 144 bits)
    # Actually, we'll compute popcount on the fly with int.bit_count()
    
    def has_perfect_matching(m, L):
        """
        Check if (m, m+L] has a perfect matching for distinct multiples.
        Left nodes: 1..12
        Right nodes: 1..L
        Edge (i, d) exists if (m+d) % i == 0, i.e., d % i == (-m) % i.
        """
        if L < n:
            return False
        
        # Build adjacency bitmasks for each left node i
        adj = [0] * n
        counts = [0] * n
        
        for i in range(1, n + 1):
            rem = (-m) % i
            mask = valid_masks[i][rem]
            limit_mask = (1 << (L + 1)) - 2
            adj[i - 1] = mask & limit_mask
            counts[i - 1] = adj[i - 1].bit_count()
            
            # Early exit if no options
            if adj[i - 1] == 0:
                return False
        
        # Check Hall's condition for small subsets as a quick prune
        # If any subset of left nodes has fewer neighbors than its size, no matching
        # For 12 nodes, checking all subsets is 2^12 = 4096, which is fast
        # But we can do a simpler check: if the union of any k left nodes' adjacencies
        # has size < k, fail. We'll check k=1 (already done) and k=2.
        
        # Quick check for pairs
        for a in range(n):
            for b in range(a + 1, n):
                union = adj[a] | adj[b]
                if union.bit_count() < 2:
                    return False
        
        # Bipartite Matching using DFS with bitmasks
        matchR = [-1] * (L + 1)
        
        def dfs(i, seen):
            avail = adj[i] & ~seen
            while avail:
                lsb = avail & -avail
                d = lsb.bit_length() - 1
                avail ^= lsb
                seen |= lsb
                
                if matchR[d] == -1 or dfs(matchR[d], seen):
                    matchR[d] = i
                    return True
            return False
        
        # Order left nodes by number of available options (popcount) - most constrained first
        order = sorted(range(n), key=lambda idx: counts[idx])
        
        for orig_idx in order:
            seen = 0
            if not dfs(orig_idx, seen):
                return False
        return True
    
    def find_max_L(m):
        """Find the maximum L such that has_perfect_matching(m, L) is true.
        We search from max_L down to n, returning the first L that works.
        This is the actual f(12,m) value we want to maximize."""
        # Binary search for minimum L is what we want to maximize
        # So find_min_L(m) = smallest L where matching exists
        # We want to maximize find_min_L(m)
        
        # Binary search for minimum L
        lo, hi = n, max_L
        min_L_for_m = max_L
        
        while lo <= hi:
            mid = (lo + hi) // 2
            if has_perfect_matching(m, mid):
                min_L_for_m = mid
                hi = mid - 1
            else:
                lo = mid + 1
        return min_L_for_m
    
    best_m = 0
    best_L = 0
    
    # Strategy: Scan all m in the period, but use a quick upper bound to skip
    # For a given m, the maximum possible L is bounded by the sum of the "gaps"
    # For each i, the multiples of i in (m, m+L] are at positions d where d % i == (-m) % i
    # The density of valid d's for node i is 1/i
    # A simple upper bound: for the most constrained node (largest i with fewest options),
    # we need at least i options, so L must be at least i * (something)
    
    # Actually, let's just scan all m but with early termination on time
    # And use the binary search efficiently
    
    for m in range(period):
        if time.time() - start_time > deadline:
            break
        
        # Quick upper bound: for each i, the number of valid d's in 1..max_L is roughly max_L/i
        # The minimum over i of (max_L / i) gives a rough bound
        # But more precisely: for node i, valid d's are d where d % i == (-m) % i
        # In 1..max_L, there are floor(max_L / i) or ceil(max_L / i) such d's
        # For a perfect matching to exist at L, each node i needs at least 1 valid d in 1..L
        # The most constrained is i=12: needs d % 12 == (-m) % 12
        # In 1..L, the count is roughly L/12
        # For L to work, we need L/12 >= 1, so L >= 12 (trivial)
        # But for the matching to exist, we need enough total capacity
        
        # A better quick check: compute the total number of valid (i,d) pairs
        # If this is < 12, no matching possible
        # Actually, we need a perfect matching, so we need at least 12 distinct d's total
        # But each i can use different d's, so the constraint is more subtle
        
        # Let's just do the binary search
        min_L_for_m = find_max_L(m)
        
        if min_L_for_m > best_L:
            best_L = min_L_for_m
            best_m = m
            
            # If we hit the theoretical max, stop
            if best_L >= max_L:
                break
    
    if best_m < 0 or best_m > 1000000:
        best_m = 0
    
    return int(best_m)
