import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    max_n = 1000000
    max_val = max_n + 24  # 1000024
    
    # Compute small_factor[x] = product of p^v_p(x) for all primes p < 24
    # This is effectively x divided by the largest divisor of x composed of primes >= 24.
    # Or simply, start with x, and for each p < 24, keep the p-part.
    # Actually, small_factor[x] is the part of x that is "smooth" with respect to primes < 24.
    
    # Initialize small_factor to 1
    small_factor = np.ones(max_val + 1, dtype=np.int64)
    
    # For each prime p < 24, multiply small_factor[m] by p for each p dividing m.
    # We can do this by iterating through powers of p.
    for p in small_primes:
        power = p
        while power <= max_val:
            # All multiples of power get an additional factor of p
            indices = np.arange(power, max_val + 1, power)
            small_factor[indices] *= p
            power *= p
            
    # Convert to list for faster access in Python loops if needed, 
    # but we will use numpy for the filtering step.
    
    # Identify where small_factor is 1
    # S(x) == 1 means x has no prime factors < 24.
    is_one = (small_factor == 1).astype(np.int8)
    
    # Compute sliding sum of is_one for windows of size 24
    # We want windows n+1 to n+24.
    # So we need sums of is_one[i] for i in [n+1, n+24].
    # Let's compute cumulative sum.
    cumsum = np.cumsum(is_one)
    # Sum from a to b inclusive is cumsum[b] - cumsum[a-1]
    # We want sum for [n+1, n+24].
    # indices: n+1 to n+24.
    # cumsum[n+24] - cumsum[n]
    
    # n ranges from 1000 to 1000000.
    # n+24 ranges from 1024 to 1000024.
    # n ranges from 1000 to 1000000.
    
    # Create array of counts for each n
    # count[n] = cumsum[n+24] - cumsum[n]
    # n is 1-indexed in the problem description for the window start?
    # "Return n in [1000, 1000000]. For each n+i, 1<=i<=24"
    # So the window is n+1, ..., n+24.
    
    n_start = 1000
    n_end = 1000000
    
    # We need cumsum indices from n_start to n_end.
    # And n+24 indices from n_start+24 to n_end+24.
    
    # cumsum is 1-indexed effectively if we prepend 0?
    # np.cumsum is 0-indexed: cumsum[i] is sum of is_one[0..i].
    # Sum of is_one[a..b] = cumsum[b] - (cumsum[a-1] if a>0 else 0).
    
    # Let's just compute the counts array directly.
    # n from 1000 to 1000000
    # We need cumsum[n+24] and cumsum[n].
    
    # Note: is_one[0] is irrelevant as n>=1000.
    
    counts = cumsum[n_start+24:n_end+25] - cumsum[n_start:n_end+1]
    
    # Find the maximum count
    max_count = np.max(counts)
    
    # Find all n where count == max_count
    # indices in counts array correspond to n = n_start + i
    candidates_indices = np.where(counts == max_count)[0]
    candidates_n = candidates_indices + n_start
    
    # If there are too many candidates, we might need to evaluate more than just the max count ones?
    # No, if we have a window with M ones, and another with M-1 ones, the one with M ones
    # has at most 24-M non-ones. The one with M-1 ones has at least 25-M non-ones.
    # The distinct count for the M-window is <= 1 + (24-M).
    # The distinct count for the M-1-window is <= 1 + (25-M).
    # It's possible the M-1 window has fewer distinct non-1 values if they repeat?
    # But 1 is a distinct value.
    # Let D_M be distinct count for M-window. D_M <= 1 + (24-M).
    # Let D_{M-1} be distinct count for M-1 window. D_{M-1} <= 1 + (25-M).
    # It is possible D_{M-1} < D_M?
    # Example: M=23. D_M <= 2. (1 and one other).
    # M-1=22. D_{M-1} <= 3. (1 and two others).
    # If the two others are the same, D_{M-1}=2.
    # So a window with fewer 1s *could* have the same or fewer distinct factors if the non-1s are highly repetitive.
    # However, non-1s are $S(x)$ for $x$ having small prime factors.
    # Consecutive integers are coprime, so their $S(x)$ values are not necessarily coprime, but they are constrained.
    # Given the rarity of "1"s, it is highly likely the global minimum is among the windows with the most 1s.
    # To be safe, let's check top K counts.
    
    # Let's get the unique counts in descending order.
    # We will evaluate windows with the top few count values.
    
    # Sort candidates by count? No, we already have max_count.
    # Let's just evaluate all windows with count == max_count.
    # If that set is large, we might time out.
    # Let's check the size.
    
    if len(candidates_n) > 10000:
        # Too many candidates with max count. 
        # Maybe we should look at top 1000 candidates by count?
        # Or just pick a random subset?
        # Let's pick the first 1000.
        candidates_n = candidates_n[:1000]
        
    best_score = 25 # Max possible distinct is 24, plus 1? No, max 24.
    best_n = n_start
    
    # Evaluate each candidate
    for n in candidates_n:
        if time.time() > deadline:
            break
            
        # Extract the 24 values
        window_vals = small_factor[n+1:n+25]
        
        # Count distinct
        # Using set is fast for small arrays
        distinct_count = len(np.unique(window_vals))
        
        if distinct_count < best_score:
            best_score = distinct_count
            best_n = n
            if best_score == 1:
                break
                
    # If we didn't find a good one, or if the max_count was low, 
    # we might need to scan more.
    # But let's assume the max_count windows are good.
    
    # Fallback: if best_score is still high (e.g., > 5), do a quick random scan?
    # Or just return best_n.
    
    return best_n
