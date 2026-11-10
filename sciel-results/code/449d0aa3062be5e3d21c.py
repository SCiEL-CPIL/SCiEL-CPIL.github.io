"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import isqrt

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    def primes_upto(n):
        """Sieve of Eratosthenes."""
        if n < 2:
            return []
        sieve = np.ones(n + 1, dtype=bool)
        sieve[0] = sieve[1] = False
        for i in range(2, isqrt(n) + 1):
            if sieve[i]:
                sieve[i*i::i] = False
        return [int(i) for i in np.where(sieve)[0]]
    
    # Generate primes up to 10000
    primes = primes_upto(10000)
    m = len(primes)
    
    if m < 2:
        return primes
    
    # DP with Pareto frontiers
    # dp[j] = sorted list of (last_gap, length) tuples, Pareto-optimal
    # meaning: increasing gap implies strictly increasing length
    # parent[j] = dict mapping (gap, length) -> (prev_index, prev_gap) for reconstruction
    # Actually, we need parent per state. Let's store:
    # dp[j] = list of (gap, length) sorted by gap, Pareto
    # parent[j] = list of (prev_index, prev_gap) parallel to dp[j]
    
    dp = [[] for _ in range(m)]
    parent = [[] for _ in range(m)]
    
    # Base case: sequence of length 1 ending at any prime
    for j in range(m):
        dp[j] = [(0, 1)]
        parent[j] = [(-1, 0)]
    
    best_len = 1
    best_end = 0
    best_gap = 0
    
    for j in range(1, m):
        if time.time() - t0 > DEADLINE:
            break
            
        candidates = []
        pj = primes[j]
        
        for i in range(j):
            d = pj - primes[i]
            lst = dp[i]
            if not lst:
                continue
            
            # Binary search for rightmost entry with gap <= d
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
                prev_gap = lst[best_idx][0]
                prev_len = lst[best_idx][1]
                candidates.append((d, prev_len + 1, i, prev_gap))
        
        if not candidates:
            # Only the base case (length 1)
            dp[j] = [(0, 1)]
            parent[j] = [(-1, 0)]
            continue
        
        # Sort by gap, then build Pareto frontier (keep only strictly increasing lengths)
        candidates.sort(key=lambda x: x[0])
        frontier = []
        parent_frontier = []
        max_len = -1
        for gap, length, prev_i, prev_gap in candidates:
            if length > max_len:
                frontier.append((gap, length))
                parent_frontier.append((prev_i, prev_gap))
                max_len = length
        
        # Merge with base case: if frontier doesn't start with (0,1), add it
        # Actually base case is always (0,1). If frontier[0] has gap > 0, we should include (0,1)
        # But (0,1) means length 1 with gap 0. Any candidate has length >= 2.
        # So (0,1) is always Pareto-optimal (smallest gap, smallest length but no other has smaller gap)
        # Wait: (0,1) vs (d, L) where d > 0 and L >= 2. 
        # (0,1) is better in gap, (d,L) is better in length. Both are Pareto-optimal.
        # So we need to include (0,1) if it's not dominated.
        # (0,1) is dominated only if there exists (g', l') with g' <= 0 and l' >= 1.
        # Since gaps are >= 0, g' = 0 and l' >= 1. If frontier has (0, L) with L >= 1, then (0,1) is dominated if L >= 1.
        # But our candidates all have gap >= 2 (since primes[j] - primes[i] >= 2 for i < j, j >= 1).
        # Actually primes[j] - primes[i] can be 2 (e.g., 3-1 no, 5-3=2). So gap can be 2.
        # (0,1) has gap 0 which is less than any candidate gap. So (0,1) is never dominated by candidates.
        # We should prepend (0,1) to the frontier.
        
        # Check if frontier already has gap 0
        if not frontier or frontier[0][0] > 0:
            frontier = [(0, 1)] + frontier
            parent_frontier = [(-1, 0)] + parent_frontier
        elif frontier[0][0] == 0:
            # frontier[0] has gap 0. If its length > 1, it dominates (0,1).
            # If its length == 1, it's the same.
            if frontier[0][1] < 1:
                frontier[0] = (0, 1)
                parent_frontier[0] = (-1, 0)
        
        dp[j] = frontier
        parent[j] = parent_frontier
        
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
    if best_len <= 1:
        return [primes[0]]
    
    seq = [primes[best_end]]
    cur_idx = best_end
    cur_gap = best_gap
    
    while cur_idx != -1:
        # Find the state (cur_gap, ?) in dp[cur_idx]
        # We need to find the index in dp[cur_idx] where gap == cur_gap
        lst = dp[cur_idx]
        par_lst = parent[cur_idx]
        
        # Binary search for cur_gap in lst
        lo, hi = 0, len(lst) - 1
        found_idx = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if lst[mid][0] == cur_gap:
                found_idx = mid
                break
            elif lst[mid][0] < cur_gap:
                lo = mid + 1
            else:
                hi = mid - 1
        
        if found_idx < 0:
            # Shouldn't happen, but break to avoid infinite loop
            break
        
        prev_i, prev_gap = par_lst[found_idx]
        if prev_i == -1:
            break
        seq.append(primes[prev_i])
        cur_idx = prev_i
        cur_gap = prev_gap
    
    seq.reverse()
    return seq

# EVOLVE-BLOCK-END
