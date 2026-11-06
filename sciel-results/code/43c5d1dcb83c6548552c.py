import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 70.0

    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    primes = [p for p in range(2, limit + 1) if is_prime[p]]
    n = len(primes)
    m = n - 1
    gaps = [primes[i + 1] - primes[i] for i in range(m)]
    
    prefix = [0] * (m + 1)
    for i in range(m):
        prefix[i + 1] = prefix[i] + gaps[i]
    
    max_sum = prefix[m]  # total sum of all gaps = primes[-1] - primes[0]
    
    # Fenwick tree for prefix max, indexed by segment sum (1-indexed)
    # We'll use size max_sum + 2
    size = max_sum + 2
    bit = [0] * (size + 1)  # stores max length
    bit_prev_end = [-1] * (size + 1)  # stores which prev_end gave the max
    bit_state_idx = [-1] * (size + 1)  # stores which state index
    
    def bit_update(idx, val, pe, si):
        while idx <= size:
            if val > bit[idx]:
                bit[idx] = val
                bit_prev_end[idx] = pe
                bit_state_idx[idx] = si
            idx += idx & (-idx)
    
    def bit_query(idx):
        # returns (max_len, prev_end, state_idx) for sums in [1, idx]
        best = 0
        best_pe = -1
        best_si = -1
        while idx > 0:
            if bit[idx] > best:
                best = bit[idx]
                best_pe = bit_prev_end[idx]
                best_si = bit_state_idx[idx]
            idx -= idx & (-idx)
        return best, best_pe, best_si
    
    # dp[end] will store the best state for reconstruction
    # We need: for each end, the best (length, seg_sum, prev_end, prev_state_idx)
    best_at_end = [None] * m  # (length, seg_sum, prev_end, prev_state_idx)
    
    best_len = 1
    best_end = 0
    best_state_idx = 0
    
    for end in range(m):
        if time.time() > deadline:
            break
        
        p_end = prefix[end + 1]
        best_here = (1, p_end, -1, -1)  # single segment from 0 to end
        
        for prev_end in range(end):
            seg_sum = p_end - prefix[prev_end + 1]
            # Query BIT for max length among all states with sum <= seg_sum
            q_len, q_pe, q_si = bit_query(seg_sum)
            if q_len >= 1:
                cand_len = q_len + 1
                if cand_len > best_here[0]:
                    best_here = (cand_len, seg_sum, q_pe, q_si)
        
        # Also consider starting fresh (length 1)
        if best_here[0] < 1:
            best_here = (1, p_end, -1, -1)
        
        best_at_end[end] = best_here
        
        # Update BIT at position seg_sum with the new state
        s = best_here[1]
        L = best_here[0]
        bit_update(s, L, end, 0)
        
        if L > best_len:
            best_len = L
            best_end = end
            best_state_idx = 0
    
    # Reconstruct
    segment_ends = []
    end = best_end
    state_idx = best_state_idx
    while end != -1:
        L, S, prev_end, prev_state_idx = best_at_end[end]
        segment_ends.append(end)
        end = prev_end
        state_idx = prev_state_idx
    
    segment_ends.reverse()
    
    prime_indices = [0]
    for se in segment_ends:
        prime_indices.append(se + 1)
    
    result = [primes[i] for i in prime_indices]
    return result
