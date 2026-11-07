import time
import sys

def construct():
    start_time = time.time()
    deadline = 80.0 # 90s limit, keep margin
    
    N = 12
    LCM = 27720
    
    # Precompute indices sorted by "tightness"
    # Tightness is roughly 1/i, but depends on L.
    # We can just sort by i descending? No, divisibility by larger i is rarer.
    # So check larger i first? 
    # Actually, in backtracking, we should pick the variable with the fewest choices first.
    # The number of choices for i is ~ L/i.
    # So larger i has fewer choices.
    # So we should assign larger i first?
    # Wait, if we assign i=12, we pick b divisible by 12.
    # Then i=11, pick b divisible by 11.
    # This seems efficient.
    
    indices = list(range(N, 0, -1)) # 12 down to 1
    
    def can_match(m, L):
        # Candidates are m+1 ... m+L
        # We need distinct b_i for i=1..12 such that i | b_i
        
        # Quick check: for each i, is there at least one multiple in (m, m+L]?
        # If not, return False.
        for i in indices:
            # First multiple of i > m is: m + 1 + ((i - 1 - m) % i) ?
            # Let x > m. x = k*i.
            # k > m/i. k_min = floor(m/i) + 1.
            # x_min = k_min * i.
            # If x_min > m + L, no candidate.
            k_min = (m // i) + 1
            x_min = k_min * i
            if x_min > m + L:
                return False
        
        # Backtracking
        # available: set of numbers in (m, m+L]
        # Using a list of booleans or a set?
        # L is small. A set is fine.
        available = set(range(m + 1, m + L + 1))
        
        # Optimization: Pre-filter available for each i?
        # No, just check in the loop.
        
        def dfs(idx_pos, assigned_count):
            if idx_pos == N:
                return True
            
            i = indices[idx_pos]
            
            # Find candidates for i in available
            # Candidates must be divisible by i
            candidates = []
            for b in available:
                if b % i == 0:
                    candidates.append(b)
            
            if not candidates:
                return False
            
            for b in candidates:
                available.remove(b)
                if dfs(idx_pos + 1, assigned_count + 1):
                    return True
                available.add(b)
                
            return False
        
        return dfs(0, 0)

    best_m = 0
    best_L = 0
    
    for m in range(LCM):
        if time.time() > deadline:
            break
            
        # Binary search for min L
        lo, hi = 1, 300 # Upper bound guess
        # Ensure hi is valid? If not, increase.
        # But max L is small.
        
        # We want the MIN L such that can_match is True.
        # If can_match(1) is True, L=1.
        # We want to maximize this min L.
        
        # Actually, we can just increment L?
        # Binary search is better.
        
        # Check if hi is sufficient
        while not can_match(m, hi):
            hi *= 2
            if hi > 1000: break # Safety
            
        # Now find min L in [lo, hi]
        # Standard binary search for first True
        while lo < hi:
            mid = (lo + hi) // 2
            if can_match(m, mid):
                hi = mid
            else:
                lo = mid + 1
        
        L_min = lo
        
        if L_min > best_L:
            best_L = L_min
            best_m = m
            
    return best_m
