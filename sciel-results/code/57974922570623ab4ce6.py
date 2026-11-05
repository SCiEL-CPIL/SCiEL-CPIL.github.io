"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    deadline = time.time() + 80.0  # safety margin before 90s kill
    
    # Generate all primes <= 10000 using Sieve of Eratosthenes
    limit = 10000
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    n = len(primes)
    
    import bisect
    
    # For each prime index i, store:
    #   dp_gaps[i] = sorted list of gap values
    #   dp_prefix[i] = prefix max lengths
    #   dp_entries[i] = list of (gap, length) sorted by gap
    #   pred[i] = dict: gap -> (prev_idx, prev_gap)
    
    dp_gaps = [None] * n
    dp_prefix = [None] * n
    dp_entries = [None] * n
    pred = [None] * n
    
    best_len = 1
    best_j = 0
    best_end_gap = 0
    
    for j in range(n):
        if time.time() > deadline:
            break
        
        candidates = {0: 1}
        preds = {0: (None, None)}
        
        for i in range(j):
            gap = primes[j] - primes[i]
            sg = dp_gaps[i]
            if sg is None:
                continue
            pos = bisect.bisect_right(sg, gap)
            if pos > 0:
                best_prev = dp_prefix[i][pos - 1]
                if best_prev > 0:
                    new_len = best_prev + 1
                    if gap not in candidates or candidates[gap] < new_len:
                        candidates[gap] = new_len
                        # Find predecessor: scan dp_entries[i] for gap' <= gap with length >= best_prev
                        entries_i = dp_entries[i]
                        # Binary search rightmost gap' <= gap
                        lo, hi = 0, len(entries_i)
                        while lo < hi:
                            mid = (lo + hi) // 2
                            if entries_i[mid][0] <= gap:
                                lo = mid + 1
                            else:
                                hi = mid
                        # Scan backwards from lo-1
                        for k in range(lo - 1, -1, -1):
                            if entries_i[k][1] >= best_prev:
                                preds[gap] = (i, entries_i[k][0])
                                break
        
        if not candidates:
            candidates = {0: 1}
            preds = {0: (None, None)}
        
        pred[j] = preds
        
        items = sorted(candidates.items())
        dp_entries[j] = items
        gaps_list = [x[0] for x in items]
        prefix = []
        cur_max = 0
        for gap, length in items:
            cur_max = max(cur_max, length)
            prefix.append(cur_max)
        
        dp_gaps[j] = gaps_list
        dp_prefix[j] = prefix
        
        max_len_j = max(candidates.values())
        if max_len_j > best_len:
            best_len = max_len_j
            best_j = j
            for gap, length in candidates.items():
                if length == max_len_j:
                    best_end_gap = gap
                    break
    
    # Backtrack
    if best_j >= 0 and best_len > 1:
        seq = [primes[best_j]]
        cur_j = best_j
        cur_gap = best_end_gap
        cur_len = best_len
        
        while cur_len > 1:
            if time.time() > deadline:
                break
            p = pred[cur_j]
            if p is None or cur_gap not in p:
                break
            prev_idx, prev_gap = p[cur_gap]
            if prev_idx is None:
                break
            seq.append(primes[prev_idx])
            cur_j = prev_idx
            cur_gap = prev_gap
            cur_len -= 1
        
        seq.reverse()
        if len(seq) >= 2:
            gaps = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
            if all(gaps[i] <= gaps[i+1] for i in range(len(gaps)-1)):
                return seq
    
    # Fallback
    values = []
    for p in primes:
        if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
            values.append(p)
    return values
# EVOLVE-BLOCK-END
