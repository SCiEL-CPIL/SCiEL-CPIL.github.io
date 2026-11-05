import time
from math import isqrt
from bisect import bisect_right

def construct():
    start_time = time.time()
    DEADLINE = 88

    def sieve(n):
        if n < 2:
            return []
        is_prime = bytearray(b'\x01') * (n + 1)
        is_prime[0] = 0
        is_prime[1] = 0
        for i in range(2, isqrt(n) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = 0
        return [p for p in range(2, n + 1) if is_prime[p]]

    primes = sieve(10000)
    n = len(primes)
    if n == 0:
        return []
    if n == 1:
        return primes

    # fronts[i] is a list of tuples (gap, length, prev_idx)
    # representing the Pareto frontier of states ending at primes[i]
    # The list is sorted by gap ascending.
    # For each gap, the length is the maximum possible for that gap or smaller.
    # Actually, the list stores non-dominated points.
    # If we sort by gap asc, lengths must be strictly increasing.
    
    fronts = [None] * n
    # State at primes[0]: gap 0, length 1, prev -1
    fronts[0] = [(0, 1, -1)]
    
    best_len = 1
    best_end = 0

    for i in range(1, n):
        if time.time() - start_time > DEADLINE:
            break

        pi = primes[i]
        candidates = []
        
        # Base case: sequence of length 1 ending at i
        # gap is 0? No, gap is defined for successive primes.
        # If length is 1, there is no previous gap.
        # We can treat "last gap" as 0 for the purpose of the first transition?
        # If we start at i, the next gap g must be >= 0.
        # So yes, (0, 1, -1) is a valid state to extend from?
        # Wait, the state (0, 1, -1) at index i means "sequence ending at i with length 1".
        # To extend from j to i, we need a state at j.
        # The state at j has a last_gap.
        # The new gap is pi - pj.
        # We need new_gap >= last_gap_at_j.
        
        # So for each j < i:
        #   gap = pi - primes[j]
        #   Find best state in fronts[j] with last_gap <= gap.
        #   New length = best_len_at_j + 1.
        #   New state at i is (gap, new_length, j).
        
        for j in range(i):
            if time.time() - start_time > DEADLINE:
                break
            front_j = fronts[j]
            if not front_j:
                continue
            
            gap = pi - primes[j]
            
            # front_j is sorted by gap.
            # We want the state with the largest gap <= current gap.
            # bisect_right returns insertion point.
            # If front_j = [(2, 5, ...), (10, 6, ...)]
            # and gap = 5.
            # bisect_right([2, 10], 5) -> 1.
            # idx = 1. best_idx = 0.
            # State is (2, 5, ...).
            
            # We need to extract gaps for bisect.
            # To avoid creating a new list of gaps every time, 
            # we could store gaps separately, but front_j is small.
            gaps_j = [s[0] for s in front_j]
            idx = bisect_right(gaps_j, gap)
            
            if idx > 0:
                best_idx = idx - 1
                prev_gap, prev_len, prev_prev = front_j[best_idx]
                new_len = prev_len + 1
                candidates.append((gap, new_len, j))
        
        if not candidates:
            # Should not happen, as j=0 always has state (0,1,-1)
            # gap = pi - p0 >= 1 >= 0.
            # So idx will be at least 1.
            fronts[i] = [(0, 1, -1)] # Fallback
            continue

        # Sort candidates by gap asc, then length desc (to handle ties in gap, prefer longer? 
        # Actually, if gap is same, we want the one with max length.
        # If we have (5, 10, j1) and (5, 12, j2), we want (5, 12, j2).
        # Sorting by (gap, -length) puts (5, 12) before (5, 10).
        candidates.sort(key=lambda x: (x[0], -x[1]))
        
        front_gaps = []
        front_lens = []
        front_prevs = []
        max_len_so_far = 0
        
        for gap, length, prev_idx in candidates:
            if length > max_len_so_far:
                front_gaps.append(gap)
                front_lens.append(length)
                front_prevs.append(prev_idx)
                max_len_so_far = length
                
        fronts[i] = list(zip(front_gaps, front_lens, front_prevs))
        
        if front_lens and front_lens[-1] > best_len:
            best_len = front_lens[-1]
            best_end = i

    # Reconstruction
    if best_len <= 1:
        return [primes[best_end]]
        
    seq = [primes[best_end]]
    idx = best_end
    current_len = best_len
    
    while current_len > 1:
        front_at_idx = fronts[idx]
        found = False
        # Find the state in front_at_idx that has length == current_len
        # And is the one that led to the current state?
        # Actually, we just need ANY state with that length.
        # But we need to trace back correctly.
        # The state at idx has length current_len.
        # It came from some prev_idx.
        # We need to find the entry in fronts[idx] with length current_len.
        # There might be multiple?
        # No, the frontier has strictly increasing lengths.
        # So there is at most one state with a specific length.
        
        for gap, length, prev_idx in front_at_idx:
            if length == current_len:
                seq.append(primes[prev_idx])
                idx = prev_idx
                current_len -= 1
                found = True
                break
        if not found:
            # This should not happen if DP is consistent
            break
            
    seq.reverse()
    
    # Verification
    if not seq:
        return [primes[0]]
        
    is_valid = True
    for k in range(len(seq) - 1):
        if seq[k] >= seq[k + 1] or seq[k] > 10000 or seq[k] < 2:
            is_valid = False
            break
            
    for k in range(len(seq)):
        p = seq[k]
        if p < 2 or p > 10000:
            is_valid = False
            break
        # Prime check
        if p > 1:
            for d in range(2, isqrt(p) + 1):
                if p % d == 0:
                    is_valid = False
                    break
        if not is_valid:
            break
            
    if is_valid:
        for k in range(2, len(seq)):
            gap_curr = seq[k] - seq[k - 1]
            gap_prev = seq[k - 1] - seq[k - 2]
            if gap_curr < gap_prev:
                is_valid = False
                break
                
    if not is_valid:
        return [primes[0]]
        
    return seq

def is_prime_check(p):
    if p < 2: return False
    if p == 2: return True
    if p % 2 == 0: return False
    for d in range(3, isqrt(p) + 1, 2):
        if p % d == 0:
            return False
    return True
