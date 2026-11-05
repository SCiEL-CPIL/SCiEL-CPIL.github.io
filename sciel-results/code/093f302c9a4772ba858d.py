import time
from math import isqrt

def construct():
    start_time = time.time()
    DEADLINE = 88

    def sieve(n):
        is_prime = bytearray(b'\x01') * (n + 1)
        if n >= 0: is_prime[0] = 0
        if n >= 1: is_prime[1] = 0
        for i in range(2, isqrt(n) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = 0
        return [p for p in range(2, n + 1) if is_prime[p]]

    primes = sieve(10000)
    n = len(primes)
    if n == 0: return []
    if n == 1: return primes

    # fronts[i] = (gaps_list, lens_list, prevs_list)
    # gaps_list is sorted ascending.
    # lens_list corresponds to gaps_list.
    # prevs_list corresponds to gaps_list.
    # Invariant: lens_list is strictly increasing.
    
    fronts = [None] * n
    # Initial state for index 0
    fronts[0] = ([0], [1], [-1])
    
    best_len = 1
    best_end = 0

    for i in range(1, n):
        if time.time() - start_time > DEADLINE:
            break

        pi = primes[i]
        
        # We will collect candidates as (gap, length, prev_idx)
        # To optimize, we can just find the best candidate for each gap?
        # No, we need the Pareto front.
        
        # Let's try to avoid the full sort if possible?
        # The candidates come from different j's.
        # gap = pi - p[j]. As j increases, gap decreases.
        
        # Let's just use the list and sort. It's the most robust.
        # But we can optimize the inner loop.
        
        candidates = []
        # Add trivial state
        candidates.append((0, 1, -1))
        
        for j in range(i):
            front_j = fronts[j]
            if not front_j: continue
            gaps_j, lens_j, prevs_j = front_j
            gap = pi - primes[j]
            
            # Binary search in gaps_j for largest <= gap
            lo, hi = 0, len(gaps_j)
            while lo < hi:
                mid = (lo + hi) // 2
                if gaps_j[mid] <= gap:
                    lo = mid + 1
                else:
                    hi = mid
            if lo > 0:
                # The best state in front_j for this gap is at index lo-1
                idx_best = lo - 1
                last_gap = gaps_j[idx_best]
                length = lens_j[idx_best]
                prev_idx = prevs_j[idx_best]
                candidates.append((gap, length + 1, j))
        
        # Sort candidates by gap asc, then length desc
        candidates.sort(key=lambda x: (x[0], -x[1]))
        
        # Build Pareto front
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
                
        fronts[i] = (front_gaps, front_lens, front_prevs)
        
        if front_lens and front_lens[-1] > best_len:
            best_len = front_lens[-1]
            best_end = i

    # Reconstruction
    if best_len <= 1:
        return [primes[best_end]]
        
    # Find the state at best_end with length best_len
    # It's the last one in the front if best_len is the max for that end
    # But best_len is global max.
    # We need to trace back.
    
    # We need to find which state in fronts[best_end] has length best_len.
    # Since lens are increasing, we can binary search or scan.
    target_front = fronts[best_end]
    if not target_front[1]:
        return [primes[best_end]]
        
    # Find index of best_len in target_front[1]
    # It should be the last one if best_len == max(target_front[1])
    # But it's possible that a different end has the global max?
    # No, best_end is the index where we updated best_len.
    # So fronts[best_end] must contain a state with length best_len.
    
    # Find the prev_idx for that state
    # Scan is O(L) which is small.
    prev_idx = -1
    for k in range(len(target_front[1])):
        if target_front[1][k] == best_len:
            prev_idx = target_front[2][k]
            break
            
    if prev_idx == -1:
        prev_idx = target_front[2][-1]
        
    seq = [primes[best_end]]
    idx = best_end
    current_len = best_len
    
    while current_len > 1:
        front_at_idx = fronts[idx]
        # Find state with length current_len
        found = False
        for k in range(len(front_at_idx[1])):
            if front_at_idx[1][k] == current_len:
                p_idx = front_at_idx[2][k]
                seq.append(primes[p_idx])
                idx = p_idx
                current_len -= 1
                found = True
                break
        if not found:
            break
            
    seq.reverse()
    
    # Verification
    if not seq:
        return [primes[0]]
        
    is_valid = True
    for k in range(len(seq) - 1):
        if seq[k] >= seq[k + 1]:
            is_valid = False
            break
        if seq[k] > 10000 or seq[k] < 2:
            is_valid = False
            break
            
    for k in range(len(seq)):
        p = seq[k]
        if p < 2:
            is_valid = False
            break
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
        # Fallback
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
