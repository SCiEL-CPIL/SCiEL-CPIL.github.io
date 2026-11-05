import time
from math import isqrt
from bisect import bisect_right

def construct():
    start_time = time.time()
    DEADLINE = 85

    def sieve(n):
        is_prime = bytearray(b'\x01') * (n + 1)
        if n >= 0:
            is_prime[0] = 0
        if n >= 1:
            is_prime[1] = 0
        for i in range(2, isqrt(n) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = 0
        return [p for p in range(2, n + 1) if is_prime[p]]

    primes = sieve(10000)
    n = len(primes)

    if n <= 1:
        return primes

    # states[i] is a list of tuples (last_gap, length, prev_idx)
    # sorted by last_gap ascending, lengths strictly increasing
    states = [None] * n
    # Virtual start: gap 0, length 0? No, length 1 for the prime itself.
    # If we start at i, len=1, gap=0.
    states[0] = [(0, 1, -1)]

    best_length = 1
    best_end = 0

    # Pre-compute primes list for fast access
    p_list = primes
    
    for i in range(1, n):
        if time.time() - start_time > DEADLINE:
            break

        pi = p_list[i]
        candidates = [(0, 1, -1)]

        for j in range(i):
            if time.time() - start_time > DEADLINE:
                break

            sj = states[j]
            if not sj:
                continue

            gap = pi - p_list[j]
            
            # Binary search for largest entry with last_gap <= gap
            # sj is sorted by gap.
            # We need the index of the first gap > gap, then step back.
            # bisect_right returns the insertion point to the right of existing entries.
            # We want the last entry <= gap.
            # So bisect_right(sj_gaps, gap) - 1.
            
            # To avoid creating a list of gaps every time, we can do manual binary search 
            # or bisect on a list comprehension (slow).
            # Manual binary search on sj is O(log K).
            lo, hi = 0, len(sj)
            while lo < hi:
                mid = (lo + hi) // 2
                if sj[mid][0] <= gap:
                    lo = mid + 1
                else:
                    hi = mid
            
            if lo > 0:
                prev_gap, prev_len, prev_idx = sj[lo - 1]
                new_len = prev_len + 1
                # Only add if this length is potentially useful.
                # We don't know the Pareto front of i yet, so we must keep all candidates
                # and filter later.
                candidates.append((gap, new_len, j))

        # Build Pareto front for i
        if candidates:
            # Sort by gap ascending, then length descending (to keep max len for same gap)
            candidates.sort(key=lambda x: (x[0], -x[1]))
            filtered = []
            max_len_so_far = 0
            for gap, length, prev_idx in candidates:
                if length > max_len_so_far:
                    filtered.append((gap, length, prev_idx))
                    max_len_so_far = length
            
            states[i] = filtered

            if filtered:
                last_len = filtered[-1][1]
                if last_len > best_length:
                    best_length = last_len
                    best_end = i

    # Reconstruct
    if best_length <= 1:
        return [p_list[best_end]]

    target_front = states[best_end]
    if not target_front:
        return [p_list[best_end]]

    # Find state with length == best_length
    prev_idx = -1
    for gap, length, p_idx in target_front:
        if length == best_length:
            prev_idx = p_idx
            break

    if prev_idx == -1:
        prev_idx = target_front[-1][2]

    seq = [p_list[best_end]]
    idx = best_end
    current_len = best_length
    
    while current_len > 1:
        front_at_idx = states[idx]
        if not front_at_idx:
            break
        found = False
        for gap, length, p_idx in front_at_idx:
            if length == current_len:
                seq.append(p_list[p_idx])
                idx = p_idx
                current_len -= 1
                found = True
                break
        if not found:
            break

    seq.reverse()

    # Verification
    def is_prime_check(p):
        if p < 2:
            return False
        if p == 2:
            return True
        if p % 2 == 0:
            return False
        for d in range(3, isqrt(p) + 1, 2):
            if p % d == 0:
                return False
        return True

    is_valid = True
    for k in range(len(seq)):
        if not is_prime_check(seq[k]) or seq[k] > 10000:
            is_valid = False
            break
    if is_valid:
        for k in range(len(seq) - 1):
            if seq[k] >= seq[k + 1]:
                is_valid = False
                break
    if is_valid:
        for k in range(2, len(seq)):
            gap_curr = seq[k] - seq[k - 1]
            gap_prev = seq[k - 1] - seq[k - 2]
            if gap_curr < gap_prev:
                is_valid = False
                break

    if not is_valid:
        return [p_list[0]]

    return seq
