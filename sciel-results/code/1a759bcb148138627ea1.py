import time
import bisect
from math import isqrt

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill

    def primes_upto(n):
        """Sieve of Eratosthenes."""
        if n < 2:
            return []
        sieve = bytearray(b'\x01') * (n + 1)
        sieve[0] = 0
        if n >= 1:
            sieve[1] = 0
        for i in range(2, isqrt(n) + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        return [i for i in range(2, n + 1) if sieve[i]]

    # Generate primes up to 10000
    primes = primes_upto(10000)
    m = len(primes)
    
    if m < 2:
        return primes

    # dp[j] is a list of (gap, length, prev_idx, prev_gap)
    # The list is sorted by gap, and lengths are strictly increasing (Pareto frontier)
    dp = [[] for _ in range(m)]
    
    # Base case: sequence of length 1 ending at primes[0]
    # Gap is 0, length 1, no previous
    dp[0] = [(0, 1, -1, 0)]
    
    best_len = 1
    best_end = 0
    best_gap = 0

    for j in range(1, m):
        candidates = []
        p_j = primes[j]
        
        for i in range(j):
            d = p_j - primes[i]
            lst = dp[i]
            if not lst:
                continue
            
            # We need the state in lst with the largest length such that gap <= d.
            # Since lst is sorted by gap and lengths are increasing,
            # we can binary search for the rightmost gap <= d.
            # The gap is the first element of the tuple.
            
            # Create a list of just gaps for bisect? Or use a key?
            # bisect_right on list of tuples compares first element.
            # We want the index of the last tuple with gap <= d.
            # bisect_right(lst, (d, float('inf'), 0, 0)) gives the insertion point.
            # The element at insertion_point - 1 is the one we want.
            
            # Optimization: Extract gaps into a separate list for faster bisect?
            # The lists are small, so tuple comparison is likely fine.
            
            # To be safe and fast, let's maintain a parallel list of gaps for each dp[i]?
            # That doubles memory but bisect is faster on list of ints.
            # Let's try with tuples first.
            
            # bisect_right returns the index after the last matching element.
            # We want the index of the last matching element.
            # If no match, idx will be 0, so we check if idx > 0.
            
            # Note: bisect on list of tuples compares lexicographically.
            # (d, inf, 0, 0) ensures we match any tuple starting with d, and since we want <= d,
            # bisect_right finds the position to insert (d, ...) to maintain order.
            # All tuples with first element < d are before.
            # All tuples with first element == d are before the insertion point of (d, inf...).
            # So the element at pos-1 has gap <= d.
            
            pos = bisect.bisect_right(lst, (d, float('inf'), 0, 0))
            if pos > 0:
                # The best candidate from i is the one at pos-1
                # Because lengths are increasing with gaps in the Pareto frontier.
                prev_gap, prev_len, prev_idx, prev_prev_gap = lst[pos-1]
                new_len = prev_len + 1
                candidates.append((d, new_len, i, prev_gap))
        
        if not candidates:
            dp[j] = [(0, 1, -1, 0)]
            continue
        
        # Build Pareto frontier for j
        # Sort by gap, then by length (descending? No, we want to keep max length for each gap,
        # and then filter to keep only increasing lengths).
        # If we sort by gap, for same gap, we want the max length.
        # Then we iterate and keep if length > max_len_so_far.
        
        # Sort by gap ascending. If gap is same, sort by length descending?
        # Actually, just sort by gap.
        candidates.sort(key=lambda x: x[0])
        
        frontier = []
        max_len = -1
        # We need to handle duplicates carefully.
        # If we have multiple candidates with same gap d, we only need the one with max length.
        # Since we iterate sorted by gap, we can group or just check.
        # If we append (d, L1) and then (d, L2) with L2 > L1, and L2 > max_len, we keep L2.
        # But if L1 was kept because L1 > max_len_prev, and L2 > L1, then L2 > max_len_prev too.
        # So simply iterating and checking `if length > max_len` works if we process all candidates for a gap?
        # No, if we process (d, 5) then (d, 10).
        # max_len is 4.
        # (d, 5): 5 > 4, keep. max_len=5.
        # (d, 10): 10 > 5, keep. max_len=10.
        # Frontier has two entries with same gap d.
        # This is valid for binary search?
        # If we binary search for d, we might pick the first one (gap d, len 5) if we use bisect_left?
        # But we use bisect_right for <= d.
        # If frontier has [(d, 5), (d, 10)], bisect_right for d returns index after (d, 10).
        # So it picks (d, 10). Correct.
        # However, having duplicate gaps in the frontier is wasteful and might confuse logic if not careful.
        # Better to deduplicate: for each gap, keep only max length.
        
        # Let's deduplicate during the frontier building.
        # Since candidates are sorted by gap:
        last_gap = -1
        best_len_for_gap = -1
        best_state_for_gap = None
        
        for gap, length, prev_i, prev_gap in candidates:
            if gap != last_gap:
                # Process previous gap group if any
                if last_gap != -1 and best_len_for_gap > max_len:
                    frontier.append((last_gap, best_len_for_gap, best_state_for_gap[2], best_state_for_gap[3]))
                    max_len = best_len_for_gap
                last_gap = gap
                best_len_for_gap = length
                best_state_for_gap = (gap, length, prev_i, prev_gap)
            else:
                if length > best_len_for_gap:
                    best_len_for_gap = length
                    best_state_for_gap = (gap, length, prev_i, prev_gap)
        
        # Process last group
        if last_gap != -1 and best_len_for_gap > max_len:
            frontier.append((last_gap, best_len_for_gap, best_state_for_gap[2], best_state_for_gap[3]))
            max_len = best_len_for_gap
            
        dp[j] = frontier
        
        # Update best
        if frontier:
            last_gap, last_len, _, _ = frontier[-1]
            if last_len > best_len:
                best_len = last_len
                best_end = j
                best_gap = last_gap
        
        # Deadline check
        if time.time() - t0 > DEADLINE:
            break

    # Reconstruct the sequence
    seq = [primes[best_end]]
    cur_idx = best_end
    cur_gap = best_gap
    
    while cur_idx != -1:
        # Find the state in dp[cur_idx] with gap == cur_gap
        # It should be there.
        # We can just iterate or binary search. Since we need exact match, and list is small, linear is fine.
        # Or we can store the parent pointer directly in a separate dict for O(1) lookup?
        # Let's just find it in the list.
        target_state = None
        for state in dp[cur_idx]:
            if state[0] == cur_gap:
                target_state = state
                break
        
        if not target_state:
            break # Should not happen
            
        _, _, prev_i, prev_gap = target_state
        if prev_i == -1:
            break
        seq.append(primes[prev_i])
        cur_idx = prev_i
        cur_gap = prev_gap
    
    seq.reverse()
    
    # Verify
    if len(seq) >= 2:
        gaps = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
        for k in range(len(gaps)-1):
            if gaps[k+1] < gaps[k]:
                # Fallback
                return [primes[0]]
                
    return seq
