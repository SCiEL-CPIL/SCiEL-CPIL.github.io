"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    import numpy as np
    from math import isqrt
    
    # Deadline guard: finish well before 90s wall-clock
    t0 = time.time()
    DEADLINE = 85.0  # safety margin
    
    def primes_upto(n):
        """Sieve of Eratosthenes."""
        if n < 2:
            return []
        sieve = np.ones(n + 1, dtype=bool)
        sieve[0] = sieve[1] = False
        for i in range(2, isqrt(n) + 1):
            if sieve[i]:
                sieve[i*i::i] = False
        return np.array([i for i, v in enumerate(sieve) if v], dtype=np.int32)
    
    # Generate primes up to 10000
    primes = primes_upto(10000)
    m = len(primes)
    
    if m < 2:
        return [int(p) for p in primes]
    
    # DP: dp[j] = length of longest valid sequence ending at index j
    # A valid sequence has nondecreasing gaps.
    # If sequence is p[i1], p[i2], ..., p[j], then gaps g_k = p[i_{k+1}] - p[i_k] must be nondecreasing.
    # 
    # Key insight: for a subsequence, we track (last_prime, last_gap).
    # dp[j][g] would be too large. Instead, we note that the constraint is on gaps.
    # 
    # Alternative: dp[j] where we also need to know the previous gap.
    # Let's use: dp[j] = max length ending at primes[j], and prev_gap[j] = the gap used to reach the optimal.
    # But this greedy doesn't work because a shorter sequence with a smaller last gap might extend further.
    #
    # Correct approach: For each pair (i, j) with i < j, we can extend a sequence ending at i with gap g = primes[j]-primes[i]
    # if the previous gap at i was <= g.
    #
    # Let dp[i][g_index] be too complex. Instead:
    # f(j, last_gap) = max length of valid sequence ending at primes[j] with last gap = last_gap.
    # But last_gap can be up to 10000, and j up to 1229. That's 1229*10000 ~ 12M states, each with O(m) transitions = too slow.
    #
    # Better: dp[j] = (best_length, min_last_gap_for_that_length)
    # No, that's not sufficient either.
    #
    # Let's think differently. For each j, consider all i < j. The gap is d = primes[j] - primes[i].
    # We want the longest sequence ending at i whose last gap <= d.
    # Let best[i][d] = max length of valid sequence ending at i with last gap exactly d? No.
    #
    # Let's define: for each i, we store a list of (gap, length) pairs representing the Pareto frontier:
    # among all valid sequences ending at i, we keep (last_gap, length) where no other has both <= gap and >= length.
    #
    # Actually, simpler: dp[i] = list of (last_gap, length) for sequences ending at primes[i].
    # For each j, for each i < j, d = primes[j] - primes[i], we look at dp[i] and find max length where last_gap <= d.
    # Then we add (d, that_length + 1) to dp[j].
    # We also maintain the Pareto frontier to keep dp[j] small.
    #
    # This is O(m^2 * log(m)) approximately if frontiers are small.
    
    # dp[j] will be a sorted list of (gap, length) tuples, Pareto-optimal (increasing gap, increasing length)
    dp = [[] for _ in range(m)]
    dp[0] = [(0, 1)]  # single element, gap 0, length 1
    
    # Track best overall
    best_len = 1
    best_end = 0
    best_gap = 0
    
    for j in range(1, m):
        candidates = []
        for i in range(j):
            d = primes[j] - primes[i]
            # Find max length in dp[i] with last_gap <= d
            # dp[i] is sorted by gap, with increasing length (Pareto)
            # Binary search for largest gap <= d
            lst = dp[i]
            if not lst:
                continue
            # Find rightmost index where gap <= d
            lo, hi = 0, len(lst) - 1
            best_idx = -1
            while lo <= hi:
                mid = (lo + hi) // 2
                if lst[mid][0] <= d:
                    best_idx = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            if best_idx >= 0:
                candidates.append((d, lst[best_idx][1] + 1))
        
        if not candidates:
            dp[j] = [(0, 1)]
            if 1 > best_len:
                best_len = 1
                best_end = j
                best_gap = 0
            continue
        
        # Pareto frontier: sort by gap, keep only increasing length
        candidates.sort(key=lambda x: x[0])
        frontier = []
        max_len = -1
        for gap, length in candidates:
            if length > max_len:
                frontier.append((gap, length))
                max_len = length
        
        dp[j] = frontier
        
        # Update best
        if frontier:
            last_gap, last_len = frontier[-1]
            if last_len > best_len:
                best_len = last_len
                best_end = j
                best_gap = last_gap
            elif last_len == best_len and last_gap < best_gap:
                best_gap = last_gap
    
    # Reconstruct the sequence
    if best_len == 1:
        return [int(primes[0])]
    
    # We need to trace back. Let's store parent info.
    # Actually, let's redo with parent tracking.
    
    # Clear and redo with parent tracking
    dp2 = [[] for _ in range(m)]
    parent = [(-1, -1) for _ in range(m)]  # (prev_index, gap_used)
    dp2[0] = [(0, 1)]
    
    best_len2 = 1
    best_end2 = 0
    best_gap2 = 0
    
    for j in range(1, m):
        candidates = []
        for i in range(j):
            d = primes[j] - primes[i]
            lst = dp2[i]
            if not lst:
                continue
            lo, hi = 0, len(lst) - 1
            best_idx = -1
            while lo <= hi:
                mid = (lo + hi) // 2
                if lst[mid][0] <= d:
                    best_idx = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            if best_idx >= 0:
                candidates.append((d, lst[best_idx][1] + 1, i, lst[best_idx][0]))
        
        if not candidates:
            dp2[j] = [(0, 1)]
            parent[j] = (-1, 0)
            continue
        
        candidates.sort(key=lambda x: x[0])
        frontier = []
        max_len = -1
        for gap, length, prev_i, prev_gap in candidates:
            if length > max_len:
                frontier.append((gap, length))
                # Store parent for this state
                # We need to store parent per state, but frontier is compacted.
                # Let's store parent as dict: (gap, length) -> (prev_i, prev_gap)
                max_len = length
        
        dp2[j] = frontier
        
        # For reconstruction, we need to know which (prev_i, prev_gap) led to the best state at j
        # Let's find the best state at j and trace back
        if frontier:
            last_gap, last_len = frontier[-1]
            if last_len > best_len2:
                best_len2 = last_len
                best_end2 = j
                best_gap2 = last_gap
    
    # Trace back from best_end2, best_gap2
    seq = [int(primes[best_end2])]
    cur_idx = best_end2
    cur_gap = best_gap2
    
    while cur_idx > 0:
        # Find i < cur_idx such that primes[cur_idx] - primes[i] == cur_gap
        # and the sequence ending at i with last_gap <= cur_gap has length = current_length - 1
        # We need to find the right i and prev_gap.
        
        # Search for i where d = primes[cur_idx] - primes[i] == cur_gap
        # primes[i] = primes[cur_idx] - cur_gap
        target = primes[cur_idx] - cur_gap
        # Find i such that primes[i] == target
        # Since primes is sorted, binary search
        lo, hi = 0, cur_idx - 1
        found = False
        while lo <= hi:
            mid = (lo + hi) // 2
            if primes[mid] == target:
                i = mid
                found = True
                break
            elif primes[mid] < target:
                lo = mid + 1
            else:
                hi = mid - 1
        
        if not found:
            break
        
        # Now find prev_gap: the last gap of the sequence ending at i
        # It should be <= cur_gap and the length should be current_len - 1
        # We need to determine what prev_gap was. It's the gap that was used to extend from i to cur_idx.
        # The prev_gap is the last gap of the sequence ending at i, which is <= cur_gap.
        # From dp2[i], we need to find the state with the max length that is <= current_len - 1 and gap <= cur_gap.
        # But we need the exact prev_gap to continue tracing.
        
        # The prev_gap is determined by the frontier. Let's find it.
        lst = dp2[i]
        # Find the state in lst with length == (current_length - 1) and gap <= cur_gap
        # The current length at cur_idx is what? We tracked best_len2 but need per-state.
        
        # This is getting complex. Let's use a different reconstruction approach.
        # Store for each (j, gap) the parent (i, prev_gap) explicitly.
        
        break  # Fallback: return just the last element if reconstruction fails
    
    # Simpler reconstruction: store parent pointers properly
    # Let's redo once more with full parent tracking per frontier state
    
    dp3 = [{} for _ in range(m)]  # dp3[j][gap] = (length, prev_i, prev_gap)
    dp3[0][0] = (1, -1, 0)
    
    # Also maintain Pareto frontier as sorted list for binary search
    frontiers = [[] for _ in range(m)]
    frontiers[0] = [(0, 1)]
    
    best_len3 = 1
    best_end3 = 0
    best_gap3 = 0
    
    for j in range(1, m):
        for i in range(j):
            d = primes[j] - primes[i]
            # Find max length in frontiers[i] with gap <= d
            fl = frontiers[i]
            if not fl:
                continue
            lo, hi = 0, len(fl) - 1
            best_idx = -1
            while lo <= hi:
                mid = (lo + hi) // 2
                if fl[mid][0] <= d:
                    best_idx = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            if best_idx >= 0:
                prev_gap, prev_len = fl[best_idx]
                new_len = prev_len + 1
                # Check if this is better than existing dp3[j].get(d)
                if d not in dp3[j] or new_len > dp3[j][d][0]:
                    dp3[j][d] = (new_len, i, prev_gap)
        
        # Build Pareto frontier for j
        if dp3[j]:
            items = [(gap, info[0]) for gap, info in dp3[j].items()]
            items.sort(key=lambda x: x[0])
            frontier = []
            max_len = -1
            for gap, length in items:
                if length > max_len:
                    frontier.append((gap, length))
                    max_len = length
            frontiers[j] = frontier
            
            # Update best
            if frontier:
                last_gap, last_len = frontier[-1]
                if last_len > best_len3:
                    best_len3 = last_len
                    best_end3 = j
                    best_gap3 = last_gap
    
    # Reconstruct
    seq = [int(primes[best_end3])]
    cur_idx = best_end3
    cur_gap = best_gap3
    
    while cur_idx != -1:
        if cur_gap in dp3[cur_idx]:
            length, prev_i, prev_gap = dp3[cur_idx][cur_gap]
            if prev_i == -1:
                break
            seq.append(int(primes[prev_i]))
            cur_idx = prev_i
            cur_gap = prev_gap
        else:
            break
    
    seq.reverse()
    
    # Verify and return
    if len(seq) >= 1:
        return seq
    else:
        return [int(primes[0])]

# EVOLVE-BLOCK-END
