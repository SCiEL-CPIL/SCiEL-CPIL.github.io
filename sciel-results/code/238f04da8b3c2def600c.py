def construct():
    import time
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    def elapsed():
        return time.time() - start_time

    from math import isqrt
    import numpy as np
    from bisect import bisect_left, bisect_right

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
    primes_arr = np.array(primes, dtype=np.int64)

    if n <= 1:
        return primes

    # DP with Pareto-optimal states
    # states[i] is a list of (last_gap, length, prev_index) sorted by last_gap ascending
    # with strictly increasing length (Pareto front)
    states = [[] for _ in range(n)]
    states[0].append((0, 1, -1))

    best_length = 1
    best_end = 0

    for i in range(1, n):
        if elapsed() > DEADLINE:
            break

        pi = primes[i]
        gaps = pi - primes_arr[:i]  # shape (i,)
        
        best_len = 1
        best_j = -1
        candidates = [(0, 1, -1)]  # standalone
        
        for j in range(i):
            if elapsed() > DEADLINE:
                break
            gap = int(gaps[j])
            sj = states[j]
            if not sj:
                continue
            # Binary search in sj (sorted by gap)
            lo, hi = 0, len(sj)
            while lo < hi:
                mid = (lo + hi) // 2
                if sj[mid][0] <= gap:
                    lo = mid + 1
                else:
                    hi = mid
            if lo > 0:
                last_gap_j, len_j, prev_j = sj[lo - 1]
                new_len = len_j + 1
                if new_len > best_len:
                    best_len = new_len
                    best_j = j
                candidates.append((gap, new_len, j))

        # Filter to Pareto-optimal
        if candidates:
            candidates.sort(key=lambda x: (x[0], -x[1]))
            filtered = []
            max_len = 0
            for gap, length, prev in candidates:
                if length > max_len:
                    filtered.append((gap, length, prev))
                    max_len = length
            states[i] = filtered

            if filtered:
                last_len = filtered[-1][1]
                if last_len > best_length:
                    best_length = last_len
                    best_end = i

    # Reconstruct the sequence ending at best_end
    if best_length <= 1:
        return [primes[best_end]]

    def reconstruct(end_idx):
        seq = []
        idx = end_idx
        while idx != -1:
            seq.append(primes[idx])
            sj = states[idx]
            if not sj:
                break
            # Find the state entry that gives the best length for this index
            target_len = max(e[1] for e in sj)
            prev = -1
            for gap_val, length_val, prev_val in sj:
                if length_val == target_len:
                    prev = prev_val
                    break
            idx = prev
        seq.reverse()
        return seq

    seq = reconstruct(best_end)

    # Local search improvements
    if elapsed() < DEADLINE - 5:
        prime_to_idx = {p: i for i, p in enumerate(primes)}
        
        def greedy_extend_smallest(start_seq):
            """Greedy extension: always pick the smallest valid next prime."""
            extended = list(start_seq)
            while elapsed() < DEADLINE - 5:
                if len(extended) >= 2:
                    last_gap = extended[-1] - extended[-2]
                else:
                    last_gap = 0
                last_prime = extended[-1]
                target = last_prime + last_gap
                pos = bisect_left(primes, target)
                if pos < len(primes) and primes[pos] >= target:
                    extended.append(primes[pos])
                else:
                    break
            return extended

        def greedy_extend_largest(start_seq):
            """Greedy extension: pick the largest valid next prime."""
            extended = list(start_seq)
            while elapsed() < DEADLINE - 5:
                if len(extended) >= 2:
                    last_gap = extended[-1] - extended[-2]
                else:
                    last_gap = 0
                last_prime = extended[-1]
                target = last_prime + last_gap
                pos = bisect_left(primes, target)
                if pos < len(primes):
                    if 9973 >= target:
                        extended.append(9973)
                    else:
                        break
                else:
                    break
            return extended

        def greedy_prepend_smallest(start_seq):
            """Greedy prepend: pick the smallest valid previous prime."""
            extended = list(start_seq)
            while elapsed() < DEADLINE - 5:
                if len(extended) >= 2:
                    first_gap = extended[1] - extended[0]
                else:
                    first_gap = 0
                first_prime = extended[0]
                max_p = first_prime - first_gap
                pos = bisect_right(primes, max_p) - 1
                if pos >= 0:
                    extended.insert(0, primes[pos])
                else:
                    break
            return extended

        def greedy_prepend_largest(start_seq):
            """Greedy prepend: pick the largest valid previous prime."""
            extended = list(start_seq)
            while elapsed() < DEADLINE - 5:
                if len(extended) >= 2:
                    first_gap = extended[1] - extended[0]
                else:
                    first_gap = 0
                first_prime = extended[0]
                max_p = first_prime - first_gap
                if max_p >= 2:
                    extended.insert(0, 2)
                else:
                    break
            return extended

        def try_insertion(seq_candidate):
            """Try to insert primes into the sequence to increase length."""
            extended = list(seq_candidate)
            improved = True
            while improved and elapsed() < DEADLINE - 5:
                improved = False
                new_extended = list(extended)
                for pos in range(len(extended) + 1):
                    if elapsed() > DEADLINE - 5:
                        break
                    if pos == 0:
                        min_gap = 0
                        max_prime = extended[0] - 1 if extended else 10000
                    elif pos == len(extended):
                        min_gap = 0
                        if len(extended) >= 2:
                            min_gap = extended[-1] - extended[-2]
                        max_prime = 10000
                    else:
                        left_gap = extended[pos] - extended[pos - 1] if pos > 0 else 0
                        right_gap = extended[pos + 1] - extended[pos] if pos + 1 < len(extended) else 0
                        prev_gap = extended[pos - 1] - extended[pos - 2] if pos >= 2 else 0
                        next_gap = extended[pos + 1] - extended[pos] if pos + 1 < len(extended) else 10000
                        a = extended[pos - 1]
                        b = extended[pos]
                        lo_p = max(a + prev_gap, b - next_gap, a + 1)
                        hi_p = (a + b) // 2
                        if lo_p <= hi_p:
                            pidx = bisect_left(primes, lo_p)
                            if pidx < len(primes) and primes[pidx] <= hi_p:
                                new_extended.insert(pos, primes[pidx])
                                improved = True
                                break
                if improved:
                    extended = new_extended
            return extended

        best_seq = list(seq)
        best_len = len(seq)

        # Try all four greedy directions from the current best
        for ext_fn in [greedy_extend_smallest, greedy_extend_largest,
                       greedy_prepend_smallest, greedy_prepend_largest]:
            ext = ext_fn(seq)
            if len(ext) > best_len:
                best_len = len(ext)
                best_seq = ext

        # Try insertion polish on the best
        if elapsed() < DEADLINE - 10:
            polished = try_insertion(best_seq)
            if len(polished) > best_len:
                best_len = len(polished)
                best_seq = polished

        # Try other candidate endpoints
        if elapsed() < DEADLINE - 15:
            candidates_to_try = []
            for i in range(n):
                sj = states[i]
                if sj:
                    max_l = sj[-1][1]
                    if max_l >= best_len - 3:
                        candidates_to_try.append((max_l, i))
            candidates_to_try.sort(key=lambda x: -x[0])
            tried = 0
            for max_l, i in candidates_to_try:
                if elapsed() > DEADLINE - 10:
                    break
                if tried >= 30:
                    break
                seq_i = reconstruct(i)
                tried += 1
                for ext_fn in [greedy_extend_smallest, greedy_extend_largest,
                               greedy_prepend_smallest, greedy_prepend_largest]:
                    ext = ext_fn(seq_i)
                    if len(ext) > best_len:
                        best_len = len(ext)
                        best_seq = ext
                if elapsed() < DEADLINE - 10:
                    polished = try_insertion(best_seq)
                    if len(polished) > best_len:
                        best_len = len(polished)
                        best_seq = polished

        if len(best_seq) > len(seq):
            seq = best_seq

    return seq
