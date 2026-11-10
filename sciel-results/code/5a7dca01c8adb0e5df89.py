import time
from math import isqrt

def construct():
    t0 = time.time()
    DEADLINE = 85.0

    def primes_upto(n):
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

    primes = primes_upto(10000)
    m = len(primes)
    
    if m < 2:
        return primes

    # dp[j] = list of (gap, length) Pareto frontier
    # par[j] = list of (prev_i, prev_gap) corresponding to dp[j]
    dp = [[] for _ in range(m)]
    par = [[] for _ in range(m)]
    
    dp[0] = [(0, 1)]
    par[0] = [(-1, 0)]
    
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
            
            # Binary search for rightmost gap <= d
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
                prev_gap, prev_len = lst[best_idx]
                new_len = prev_len + 1
                # Store candidate: (gap, length, prev_i, prev_gap, prev_idx_in_dp_i)
                # We need prev_idx_in_dp_i to get the correct parent from par[i]
                candidates.append((d, new_len, i, prev_gap, best_idx))
        
        if not candidates:
            dp[j] = [(0, 1)]
            par[j] = [(-1, 0)]
            continue
        
        # Sort by gap, then by length descending (to pick best length for same gap if needed, 
        # but Pareto frontier keeps strictly increasing length, so for same gap we keep max length)
        candidates.sort(key=lambda x: (x[0], -x[1]))
        
        frontier = []
        frontier_par = []
        max_len = -1
        
        for gap, length, prev_i, prev_gap, prev_idx in candidates:
            if length > max_len:
                frontier.append((gap, length))
                # Retrieve parent info from par[prev_i] at index prev_idx
                frontier_par.append(par[prev_i][prev_idx])
                max_len = length
        
        dp[j] = frontier
        par[j] = frontier_par
        
        # Update best
        if frontier:
            last_gap, last_len = frontier[-1]
            if last_len > best_len:
                best_len = last_len
                best_end = j
                best_gap = last_gap

        if time.time() - t0 > DEADLINE:
            break

    # Reconstruct
    seq = [primes[best_end]]
    cur_idx = best_end
    cur_gap = best_gap
    
    while cur_idx != -1:
        # Find cur_gap in dp[cur_idx]
        lst = dp[cur_idx]
        lo, hi = 0, len(lst) - 1
        idx = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if lst[mid][0] == cur_gap:
                idx = mid
                break
            elif lst[mid][0] < cur_gap:
                lo = mid + 1
            else:
                hi = mid - 1
        
        if idx == -1:
            # Should not happen if logic is correct
            break
            
        prev_i, prev_gap = par[cur_idx][idx]
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
                values = [primes[0]]
                for p in primes[1:]:
                    if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
                        values.append(p)
                return values

    return seq
