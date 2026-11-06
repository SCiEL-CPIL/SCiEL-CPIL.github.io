import time
import math
from bisect import bisect_left, bisect_right

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    # Sieve of Eratosthenes to get primes up to 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    primes = [p for p in range(2, limit + 1) if is_prime[p]]
    n = len(primes)  # ~1229 primes
    
    # Compute consecutive prime gaps
    # gaps[i] = primes[i+1] - primes[i] for i in range(n-1)
    m = n - 1  # number of gaps
    gaps = [primes[i + 1] - primes[i] for i in range(m)]
    
    # Precompute prefix sums of gaps
    # prefix[i] = sum of gaps[0..i-1] = primes[i] - primes[0]
    # So sum of gaps[a..b] = prefix[b+1] - prefix[a]
    prefix = [0] * (m + 1)
    for i in range(m):
        prefix[i + 1] = prefix[i] + gaps[i]
    
    # dp[end] = list of (length, last_segment_sum) for sequences where the last segment ends at gap index `end`
    # A segment from start to end has sum = prefix[end+1] - prefix[start]
    # We need: previous_last_segment_sum <= current_segment_sum
    
    # For reconstruction: store parent info
    # Let's use: dp[end] = list of (length, last_sum, prev_end, prev_state_idx)
    
    dp = [[] for _ in range(m)]
    
    # Base case: a single segment from 0 to end
    # length = 1 (one segment = two primes)
    for end in range(m):
        seg_sum = prefix[end + 1] - prefix[0]  # = primes[end+1] - primes[0]
        dp[end].append((1, seg_sum, -1, -1))
    
    # For each end, try to extend sequences from earlier ends
    # dp[end] should contain all non-dominated states
    # A state (L1, S1) dominates (L2, S2) if L1 >= L2 and S1 <= S2
    
    best_len = 1
    best_end = 0
    best_state_idx = 0
    
    for end in range(1, m):
        if time.time() > deadline:
            break
        
        # Start with the base case: single segment from 0 to end
        candidates = [(1, prefix[end + 1], -1, -1)]
        
        # Try to extend from each prev_end < end
        for prev_end in range(end):
            prev_seg_sum_end = prefix[prev_end + 1]  # sum of gaps[0..prev_end]
            # The next segment starts at prev_end + 1 and ends at `end`
            # Its sum = prefix[end+1] - prefix[prev_end+1]
            next_seg_sum = prefix[end + 1] - prefix[prev_end + 1]
            
            # For each state in dp[prev_end], check if we can extend
            for state_idx, (L, S, _, _) in enumerate(dp[prev_end]):
                if S <= next_seg_sum:
                    candidates.append((L + 1, next_seg_sum, prev_end, state_idx))
        
        # Prune: keep only non-dominated states
        # Sort by length desc, then by sum asc
        candidates.sort(key=lambda x: (-x[0], x[1]))
        
        non_dom = []
        min_sum_encountered = float('inf')
        for cand in candidates:
            L, S, pe, si = cand
            if S < min_sum_encountered:
                non_dom.append(cand)
                min_sum_encountered = S
        
        dp[end] = non_dom
        
        # Update global best
        if non_dom:
            current_best = non_dom[0][0]
            if current_best > best_len:
                best_len = current_best
                best_end = end
                best_state_idx = 0
    
    # Reconstruct the sequence of segment endpoints
    # Then convert to prime indices
    segment_ends = []
    end = best_end
    state_idx = best_state_idx
    while end != -1:
        L, S, prev_end, prev_state_idx = dp[end][state_idx]
        segment_ends.append(end)
        end = prev_end
        state_idx = prev_state_idx
    
    segment_ends.reverse()
    
    # segment_ends contains the end indices of each segment
    # The segments are: [0, segment_ends[0]], [segment_ends[0]+1, segment_ends[1]], ..., etc.
    # Prime indices: segment from gap index a to b corresponds to primes a to b+1
    # So the primes in the sequence are at indices: 0, segment_ends[0]+1, segment_ends[1]+1, ...
    
    prime_indices = [0]
    for se in segment_ends:
        prime_indices.append(se + 1)
    
    result = [primes[i] for i in prime_indices]
    return result
