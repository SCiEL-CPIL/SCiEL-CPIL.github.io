import math
import bisect

def construct():
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    n = len(primes)
    
    # dp[i] is a list of (gap, length) sorted by gap
    # par[i] is a dict: gap -> (prev_idx, prev_gap)
    dp = [None] * n
    par = [None] * n
    
    global_max_len = 0
    global_best_i = 0
    global_best_gap = 0
    
    for i in range(n):
        p_i = primes[i]
        candidates = [] # list of (gap, length, prev_idx, prev_gap)
        
        for j in range(i):
            p_j = primes[j]
            gap_ij = p_i - p_j
            
            # dp[j] is sorted by gap
            # Find the best state in dp[j] with gap <= gap_ij
            # dp[j] is list of (gap, len)
            # We want max len among those with gap <= gap_ij
            
            # Since dp[j] is sorted by gap, and we want max len,
            # and the Pareto frontier has increasing gaps and increasing lengths,
            # the best candidate is the one with the largest gap <= gap_ij.
            # Because for Pareto frontiers, larger gap implies larger or equal length?
            # No, Pareto frontier: (g1, l1), (g2, l2) with g1 < g2 implies l1 < l2.
            # So yes, the last element with gap <= gap_ij has the max length.
            
            if not dp[j]:
                continue
                
            # Binary search for the rightmost index with gap <= gap_ij
            # dp[j] is list of tuples (gap, len)
            # We can extract gaps or use a custom search.
            # For speed, let's assume frontier is small and use linear scan?
            # Or use bisect on a list of gaps.
            
            # To avoid creating a list of gaps every time, we can just do a linear scan
            # if the list is small.
            # Let's check average size.
            # If size is > 10, binary search is better.
            
            # Let's just use a linear scan from the end?
            # No, gaps are increasing. We want the last one <= gap_ij.
            
            # Let's use bisect_right on the list of gaps.
            # We can store dp[j] as two lists: gaps[j], lens[j]
            # This might be faster for bisect.
            
            # Let's refactor dp to store separate lists for gaps and lengths?
            # Or just use a helper.
            
            # Let's stick to linear scan for now but break early?
            # The current code breaks early.
            # The binary search avoids the loop entirely.
            
            # Let's implement binary search on the list of tuples.
            # Python's bisect doesn't support key in older versions.
            # We can use bisect_right on a list of gaps.
            
            # To be safe and fast, let's pre-extract gaps?
            # That's expensive.
            
            # Let's just do a manual binary search on the list of tuples.
            lo, hi = 0, len(dp[j])
            while lo < hi:
                mid = (lo + hi) // 2
                if dp[j][mid][0] <= gap_ij:
                    lo = mid + 1
                else:
                    hi = mid
            # lo is the first index with gap > gap_ij
            # So lo-1 is the last index with gap <= gap_ij
            if lo > 0:
                best_gap_j, best_len_j = dp[j][lo-1]
                candidates.append((gap_ij, best_len_j + 1, j, best_gap_j))
        
        if not candidates:
            # Only the prime itself
            dp[i] = [(0, 1)]
            par[i] = {0: (-1, -1)}
            if 1 > global_max_len:
                global_max_len = 1
                global_best_i = i
                global_best_gap = 0
            continue
            
        # Sort candidates by gap, then by length descending
        candidates.sort(key=lambda x: (x[0], -x[1]))
        
        cur_dp = []
        cur_par = {}
        max_len_so_far = 0
        
        for gap, length, prev_idx, prev_gap in candidates:
            if length > max_len_so_far:
                cur_dp.append((gap, length))
                cur_par[gap] = (prev_idx, prev_gap)
                max_len_so_far = length
                
        dp[i] = cur_dp
        par[i] = cur_par
        
        if max_len_so_far > global_max_len:
            global_max_len = max_len_so_far
            global_best_i = i
            global_best_gap = cur_dp[-1][0]
            
    # Reconstruct
    seq = []
    i = global_best_i
    gap = global_best_gap
    while i != -1:
        seq.append(primes[i])
        if gap == -1:
            break
        prev_idx, prev_gap = par[i][gap]
        i = prev_idx
        gap = prev_gap
        
    seq.reverse()
    return seq
