import time
import bisect
from math import isqrt

def construct():
    start_time = time.time()
    DEADLINE = 85.0

    def elapsed():
        return time.time() - start_time

    # Generate all primes <= 10000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, isqrt(n) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [p for p in range(n + 1) if is_prime[p]]

    primes = sieve(10000)
    n = len(primes)

    if n <= 1:
        return primes

    # states[i] is a list of tuples: (gap, length, prev_idx, prev_gap)
    # Sorted by gap ascending. Lengths strictly increasing.
    states = [[] for _ in range(n)]
    
    # Initialize first prime
    # gap=0, length=1, prev_idx=-1, prev_gap=0
    states[0].append((0, 1, -1, 0))
    
    best_len = 1
    best_idx = 0
    best_gap = 0

    for i in range(1, n):
        if elapsed() > DEADLINE:
            break
            
        pi = primes[i]
        candidates = [(0, 1, -1, 0)] # Start new sequence at pi
        
        for j in range(i):
            gap = pi - primes[j]
            sj = states[j]
            if not sj:
                continue
            
            # Binary search in sj for rightmost entry with gap_j <= gap
            lo, hi = 0, len(sj)
            while lo < hi:
                mid = (lo + hi) // 2
                if sj[mid][0] <= gap:
                    lo = mid + 1
                else:
                    hi = mid
            
            if lo > 0:
                last_gap_j, len_j, prev_j, prev_gap_j = sj[lo - 1]
                new_len = len_j + 1
                # The new state at i comes from state (last_gap_j, ...) at j
                # The gap between j and i is 'gap'
                candidates.append((gap, new_len, j, last_gap_j))
        
        # Prune candidates to Pareto frontier
        if candidates:
            # Sort by gap, then by -length (larger length first for same gap)
            candidates.sort(key=lambda x: (x[0], -x[1]))
            
            filtered = []
            max_len_so_far = 0
            for gap, length, prev_idx, prev_gap in candidates:
                if length > max_len_so_far:
                    filtered.append((gap, length, prev_idx, prev_gap))
                    max_len_so_far = length
            
            states[i] = filtered
            
            # Update global best
            if filtered:
                last_gap, last_len, _, _ = filtered[-1]
                if last_len > best_len:
                    best_len = last_len
                    best_idx = i
                    best_gap = last_gap

    # Reconstruct the sequence
    if best_len <= 1:
        return [primes[best_idx]]

    seq = []
    idx = best_idx
    target_gap = best_gap
    
    while idx != -1:
        seq.append(primes[idx])
        sj = states[idx]
        if not sj:
            break
        
        # Find the state in sj that matches target_gap
        prev_idx = -1
        next_target_gap = 0
        found = False
        for g, l, p, pg in sj:
            if g == target_gap:
                prev_idx = p
                next_target_gap = pg
                found = True
                break
        
        if not found:
            # Fallback: if exact match not found (should not happen), break
            break
            
        idx = prev_idx
        target_gap = next_target_gap
        
    seq.reverse()
    
    return seq
