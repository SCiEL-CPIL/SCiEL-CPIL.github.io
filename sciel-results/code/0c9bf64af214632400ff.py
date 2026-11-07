import sys

def construct():
    """
    Find m in [0, 1000000] that maximizes f(12, m), where f(12, m) is the
    least positive L such that (m, m+L] contains distinct b_1,...,b_12
    with i | b_i for each i=1..12.
    """
    P = 27720  # LCM(1..12)
    MAX_L = 144
    
    # Precompute for each residue class m mod P, and for each L,
    # whether a matching exists. Since the pattern repeats every P,
    # we only need to check m in [0, P-1].
    
    def check_match(m, L):
        """Check if (m, m+L] has distinct b_1..b_12 with i|b_i."""
        # Build candidate sets for each i
        # For i in 1..12, candidates are multiples of i in (m, m+L]
        # Use bitmask representation for efficiency
        # Numbers in (m, m+L] are m+1, m+2, ..., m+L
        # We'll index them 0..L-1
        
        # For each i, find which positions (0-indexed) in the window are multiples of i
        # Position j (0-indexed) corresponds to number m+1+j
        # m+1+j is divisible by i iff (m+1+j) % i == 0
        
        cand_masks = [0] * 13  # 1-indexed
        for i in range(1, 13):
            mask = 0
            # Find first multiple of i in (m, m+L]
            start = m + 1
            if start % i == 0:
                first = start
            else:
                first = start + (i - start % i)
            
            if first > m + L:
                return False  # No candidate for i
            
            # Mark all multiples of i in the window
            for x in range(first, m + L + 1, i):
                pos = x - (m + 1)  # 0-indexed position
                if 0 <= pos < L:
                    mask |= (1 << pos)
            cand_masks[i] = mask
        
        # Now do bipartite matching via DFS with memoization
        # Sort indices by number of candidates (most constrained first)
        indices = sorted(range(1, 13), key=lambda x: bin(cand_masks[x]).count('1'))
        
        memo = {}
        
        def dfs(idx, used_mask):
            if idx == 12:
                return True
            key = (idx, used_mask)
            if key in memo:
                return memo[key]
            
            i = indices[idx]
            available = cand_masks[i] & ~used_mask
            
            while available:
                lsb = available & -available
                if dfs(idx + 1, used_mask | lsb):
                    memo[key] = True
                    return True
                available -= lsb
            
            memo[key] = False
            return False
        
        return dfs(0, 0)
    
    best_m = 0
    best_L = 0
    
    for m in range(P):
        # Find the least L such that matching exists
        # L is at least 12 (need 12 distinct numbers)
        # Binary search on L in [12, MAX_L]
        lo, hi = 12, MAX_L
        if not check_match(m, hi):
            # Should not happen, but skip
            continue
        
        while lo < hi:
            mid = (lo + hi) // 2
            if check_match(m, mid):
                hi = mid
            else:
                lo = mid + 1
        
        if lo > best_L:
            best_L = lo
            best_m = m
    
    return int(best_m)
