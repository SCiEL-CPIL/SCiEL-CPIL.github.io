import math
from functools import lru_cache

P = {'n': 12, 'maximum_start': 1000000}

def construct():
    n = P['n']
    max_m = P['maximum_start']
    
    # Precompute indices
    indices = list(range(1, n + 1))
    
    best_m = 0
    best_L = -1
    
    # Helper to check if a perfect matching exists for given m and L
    def has_matching(m, L):
        # Build candidates for each i
        # candidates[i-1] is list of values in (m, m+L] divisible by i
        candidates = []
        for i in indices:
            # First multiple > m
            first = ((m // i) + 1) * i
            if first > m + L:
                return False
            cand_list = []
            val = first
            while val <= m + L:
                cand_list.append(val)
                val += i
            candidates.append(cand_list)
        
        # Simple DFS backtracking for bipartite matching
        # Left side: indices 0..n-1 (representing i=1..n)
        # Right side: the actual values
        # Since values are large, we map them to local indices for the matching array
        # Or just use a set of used values.
        
        used_values = set()
        
        def backtrack(k):
            if k == n:
                return True
            
            # Optimization: try the index with fewest candidates first?
            # For N=12, simple order is often fine, but let's stick to fixed order for simplicity
            # or dynamic ordering if we want max speed.
            # Given the small N, fixed order is likely sufficient.
            
            i_idx = k
            i_val = indices[i_idx]
            for c in candidates[i_idx]:
                if c not in used_values:
                    used_values.add(c)
                    if backtrack(k + 1):
                        return True
                    used_values.remove(c)
            return False
        
        return backtrack(0)

    # Iterate through all possible m
    for m in range(max_m + 1):
        # Lower bound for L
        min_L = 0
        for i in indices:
            first = ((m // i) + 1) * i
            L_i = first - m
            if L_i > min_L:
                min_L = L_i
        
        # Binary search for the smallest L >= min_L such that matching exists
        # Upper bound: The problem states max 144, but let's use 200 to be safe.
        # If no match by 200, we skip (though theoretically shouldn't happen).
        lo = min_L
        hi = 200
        
        # Check if hi is even valid? If not, we might need to extend, but 200 is safe.
        if not has_matching(m, hi):
            continue
            
        # Find smallest L in [lo, hi]
        while lo < hi:
            mid = (lo + hi) // 2
            if has_matching(m, mid):
                hi = mid
            else:
                lo = mid + 1
        
        current_L = lo
        if current_L > best_L:
            best_L = current_L
            best_m = m
            
    return int(best_m)
