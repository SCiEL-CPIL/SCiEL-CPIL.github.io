"""Rewritten program: DP with Pareto frontier over last-gap for longest nondecreasing-gap prime subsequence."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

_DEADLINE = 85.0

def _sieve_primes(limit):
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    primes = _sieve_primes(10000)
    n = len(primes)
    
    # dp[j] = list of (gap, length, parent_i) Pareto frontier
    # gap: the gap used to arrive at j (0 means no prior, i.e., sequence of length 1)
    # length: sequence length ending at j
    # parent_i: index of previous prime in the sequence, or -1 if length is 1
    # Frontier sorted by gap ascending, length strictly increasing (no dominated points)
    
    dp = [[] for _ in range(n)]
    best_len = 1
    best_j = 0
    
    for j in range(n):
        if time.time() - start_time > _DEADLINE:
            break
        
        # Collect candidates: (gap, length, parent_i)
        # Base: length 1, gap 0, no parent
        candidates = [(0, 1, -1)]
        
        for i in range(j):
            gap = primes[j] - primes[i]
            frontier = dp[i]
            if not frontier:
                continue
            # Find max length in frontier where entry_gap <= gap
            # Frontier sorted by gap ascending
            lo, hi = 0, len(frontier) - 1
            best_idx = -1
            while lo <= hi:
                mid = (lo + hi) // 2
                if frontier[mid][0] <= gap:
                    best_idx = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            if best_idx >= 0:
                g, l, pi = frontier[best_idx]
                candidates.append((gap, l + 1, i))
        
        # Build Pareto frontier from candidates
        # For each gap, keep max (length, parent_i)
        gap_best = {}
        for g, l, pi in candidates:
            if g not in gap_best or l > gap_best[g][0]:
                gap_best[g] = (l, pi)
        
        sorted_items = sorted(gap_best.items())
        
        # Remove dominated: scan left to right, keep if length > max so far
        frontier = []
        max_l = 0
        for g, (l, pi) in sorted_items:
            if l > max_l:
                frontier.append((g, l, pi))
                max_l = l
        
        dp[j] = frontier
        
        if frontier:
            ml = max(f[1] for f in frontier)
            if ml > best_len:
                best_len = ml
                best_j = j
    
    # Reconstruct
    if best_len <= 1:
        return [primes[0]]
    
    seq = []
    j = best_j
    target_len = best_len
    while target_len > 0:
        seq.append(primes[j])
        # Find entry in dp[j] with length == target_len
        chosen = None
        for g, l, pi in dp[j]:
            if l == target_len:
                chosen = (g, l, pi)
                break
        if chosen is None:
            break
        _, _, pi = chosen
        if pi == -1:
            break
        j = pi
        target_len -= 1
    
    seq.reverse()
    return seq

# EVOLVE-BLOCK-END
