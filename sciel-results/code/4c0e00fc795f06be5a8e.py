"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
import bisect

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
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
    
    # DP with Pareto frontier and parent tracking
    # dp_gaps[j] = sorted list of gaps (Pareto-optimal)
    # dp_lens[j] = corresponding lengths (strictly increasing with gaps)
    # parent[j] = dict mapping gap -> (prev_index, prev_gap)
    dp_gaps = [[] for _ in range(m)]
    dp_lens = [[] for _ in range(m)]
    parent = [{} for _ in range(m)]
    
    # Base case: single element sequences (gap=0, length=1)
    dp_gaps[0] = [0]
    dp_lens[0] = [1]
    parent[0][0] = (-1, 0)
    
    best_len = 1
    best_end = 0
    best_gap = 0
    
    primes_list = primes  # local ref
    
    for j in range(1, m):
        pj = primes_list[j]
        candidates = []
        
        for i in range(j):
            d = pj - primes_list[i]
            gaps_i = dp_gaps[i]
            if not gaps_i:
                continue
            
            # Binary search for rightmost entry with gap <= d
            idx = bisect.bisect_right(gaps_i, d) - 1
            if idx >= 0:
                prev_gap = gaps_i[idx]
                prev_len = dp_lens[i][idx]
                new_len = prev_len + 1
                candidates.append((d, new_len, i, prev_gap))
        
        if not candidates:
            dp_gaps[j] = [0]
            dp_lens[j] = [1]
            parent[j][0] = (-1, 0)
            continue
        
        # Build Pareto frontier: sort by gap, keep only strictly increasing length
        candidates.sort(key=lambda x: (x[0], -x[1]))
        frontier_gaps = []
        frontier_lens = []
        max_len = -1
        for gap, length, prev_i, prev_gap in candidates:
            if length > max_len:
                frontier_gaps.append(gap)
                frontier_lens.append(length)
                parent[j][gap] = (prev_i, prev_gap)
                max_len = length
        
        dp_gaps[j] = frontier_gaps
        dp_lens[j] = frontier_lens
        
        # Update best
        if frontier_lens:
            last_gap = frontier_gaps[-1]
            last_len = frontier_lens[-1]
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
    
    while cur_idx != -1 and cur_gap in parent[cur_idx]:
        prev_i, prev_gap = parent[cur_idx][cur_gap]
        if prev_i == -1:
            break
        seq.append(primes[prev_i])
        cur_idx = prev_i
        cur_gap = prev_gap
    
    seq.reverse()
    
    # Verify the sequence has nondecreasing gaps
    if len(seq) >= 2:
        gaps = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
        for k in range(len(gaps)-1):
            if gaps[k+1] < gaps[k]:
                # Fallback to greedy
                values = [primes[0]]
                for p in primes[1:]:
                    if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
                        values.append(p)
                return values
    
    return seq

# EVOLVE-BLOCK-END
