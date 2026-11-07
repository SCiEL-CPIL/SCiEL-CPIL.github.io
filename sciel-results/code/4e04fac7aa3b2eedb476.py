import sys
sys.setrecursionlimit(10000)

# Precompute the answer for all residue classes modulo 27720
# period = lcm(1, 2, ..., 12) = 27720

def _check_matching(valid_d_lists, L):
    """
    Check if there is a perfect matching for items 0..11 using d values <= L.
    valid_d_lists[i] is a sorted list of all valid d's for item i (up to 144).
    """
    n = 12
    # Filter valid d's to be <= L
    adj = []
    for i in range(n):
        # valid_d_lists[i] is sorted
        # Find the last index <= L
        # Since L is small and lists are short, linear scan or binary search is fine.
        # Binary search:
        lo, hi = 0, len(valid_d_lists[i])
        while lo < hi:
            mid = (lo + hi) // 2
            if valid_d_lists[i][mid] <= L:
                lo = mid + 1
            else:
                hi = mid
        # lo is the count of elements <= L
        adj.append(valid_d_lists[i][:lo])
        
    # If any adj list is empty, no matching
    for lst in adj:
        if not lst:
            return False
            
    # Sort items by number of choices (fewest first) to speed up backtracking
    # We need to keep track of which original index corresponds to which adj list
    # But since we just need existence, we can sort the lists themselves.
    adj_sorted = sorted(adj, key=len)
    
    used = set()
    
    def backtrack(idx):
        if idx == n:
            return True
        for d in adj_sorted[idx]:
            if d not in used:
                used.add(d)
                if backtrack(idx + 1):
                    return True
                used.remove(d)
        return False
        
    return backtrack(0)

def _precompute():
    n = 12
    period = 27720
    max_L = 144  # upper bound on f(12, m)
    
    results = [0] * period
    
    for m in range(period):
        # Precompute valid_d_lists for this m up to max_L
        valid_d_lists = []
        for i in range(1, n + 1):
            lst = []
            start_d = (-m) % i
            if start_d == 0:
                start_d = i
            d = start_d
            while d <= max_L:
                lst.append(d)
                d += i
            valid_d_lists.append(lst)
            
        # Binary search for minimal L
        lo, hi = 1, max_L
        ans = max_L
        while lo <= hi:
            mid = (lo + hi) // 2
            if _check_matching(valid_d_lists, mid):
                ans = mid
                hi = mid - 1
            else:
                lo = mid + 1
                
        results[m] = ans
        
    return results

# Precompute at module load time
_results = _precompute()

def construct():
    # We need to find m in 0..1000000 that maximizes f(12, m)
    # f(12, m) = results[m % 27720]
    # So we want the residue r that maximizes results[r], then pick any m ≡ r (mod 27720) in [0, 1000000]
    
    period = 27720
    best_r = 0
    best_val = -1
    for r in range(period):
        if _results[r] > best_val:
            best_val = _results[r]
            best_r = r
    
    # Find the smallest m in [0, 1000000] with m ≡ best_r (mod period)
    # m = best_r is in [0, 27719], which is within [0, 1000000]
    # So m = best_r works (since best_r < 27720 <= 1000000)
    
    return best_r
