"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s
    
    n = 12
    max_m = 1000000
    
    # LCM of 1..12
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    period = 1
    for i in range(1, 13):
        period = lcm(period, i)
    # period = 27720
    
    # For each i in 1..12, precompute the set of residues r in [0, period) 
    # such that (m + r) is divisible by i, i.e., r ≡ -m (mod i)
    # Actually, for a given m, the values in (m, m+L] are m+1, m+2, ..., m+L
    # Value v = m + k (k=1..L) is divisible by i iff (m+k) % i == 0 iff k ≡ -m (mod i)
    
    # For bitmask DP: dp[mask] = True if we can assign indices corresponding to set bits in mask
    # to distinct values in (m, m+L]
    
    # Better approach: For a given m and L, check if perfect matching exists
    # Use Hall's theorem / max flow, or since n=12, use bitmask DP over indices
    
    # Let's use: for each value v in (m, m+L], determine which indices i can use it
    # Then check if we can assign each index a distinct value
    
    # Since L is at most 144 and n=12, we can use bitmask DP over the 12 indices
    # dp[mask] = set of values used? No, that's too big.
    # Instead: dp[mask] = whether we can assign the indices in mask to distinct valid values
    
    # Actually, let's think differently. For each index i, the valid values in (m, m+L] 
    # are those v where v % i == 0. The count is roughly L/i.
    # We need to check if a system of distinct representatives exists.
    
    # Use bitmask DP: process values one by one, track which indices have been assigned
    # But L can be up to 144, and 2^12 = 4096 states, so 144 * 4096 ≈ 590K operations per (m,L)
    
    # Binary search on L for each m
    
    def can_match(m, L):
        """Check if there exist distinct b_1,...,b_12 in (m, m+L] with i | b_i."""
        # For each index i (0-indexed, so i+1 is the actual index),
        # find which values in (m, m+L] are divisible by (i+1)
        # Use bitmask DP over indices
        
        # adj[i] = bitmask of values (positions 0..L-1) that index i can take
        # Then check if we can assign each index a distinct value
        
        # Alternative: dp over values. For each value v in (m, m+L], determine which indices can use it.
        # dp[mask] = True if indices in mask can be assigned to some subset of the first k values
        
        # Let's do: for each value position k (0-indexed, value = m+1+k),
        # determine which indices i can use this value (i.e., (m+1+k) % (i+1) == 0)
        # Then dp over values, tracking which indices have been assigned
        
        dp = [False] * (1 << n)
        dp[0] = True
        
        for k in range(L):
            val = m + 1 + k
            new_dp = dp[:]  # skip this value
            # For each index i that can use this value
            for i in range(n):
                if val % (i + 1) == 0:
                    bit = 1 << i
                    for mask in range(1 << n):
                        if dp[mask] and (mask & bit) == 0:
                            new_dp[mask | bit] = True
            dp = new_dp
        
        return dp[(1 << n) - 1]
    
    def f_of_m(m):
        """Compute f(12, m) = least positive L such that can_match(m, L) is True."""
        # Binary search on L
        lo, hi = 1, 144
        while lo < hi:
            mid = (lo + hi) // 2
            if can_match(m, mid):
                hi = mid
            else:
                lo = mid + 1
        return lo
    
    # Now search over m. Since f depends on m mod period, check all residues.
    # But we want m in [0, 1000000].
    
    best_m = 0
    best_f = 0
    
    # Check all residues mod period
    for residue in range(period):
        if time.time() > deadline:
            break
        m = residue
        if m > max_m:
            m = residue  # but we need m <= 1000000, and period=27720 < 1000000
        f_val = f_of_m(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
    
    # Verify best_m is in range
    if best_m > max_m:
        best_m = best_m % (max_m + 1)
    
    return int(best_m)

# EVOLVE-BLOCK-END
