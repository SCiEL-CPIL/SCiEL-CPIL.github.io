"""Rewritten program: longest increasing prime subsequence <=10000 with nondecreasing gaps."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _sieve(n):
    """Return a boolean array where index i is True if i is prime."""
    if n < 2:
        return np.zeros(n + 1, dtype=bool)
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(np.sqrt(n)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return is_prime

def construct():
    t0 = time.time()
    deadline = t0 + 80.0  # safety margin before 90s kill

    N = 10000
    is_prime = _sieve(N)
    primes = np.array([i for i in range(2, N + 1) if is_prime[i]], dtype=np.int32)
    n = len(primes)

    if n == 0:
        return []
    if n == 1:
        return [int(primes[0])]

    max_gap = int(primes[-1] - primes[0])

    # Sparse DP: for each prime index j, store a dict mapping gap -> (best_len, parent_i, parent_gap)
    # pref[j] = dict mapping gap -> prefix_max_value (max_{h<=gap} best[j,h])
    # We use lists of dicts for efficiency

    best_dicts = [None] * n       # best_dicts[j] = {gap: (length, parent_idx, parent_gap)}
    pref_dicts = [None] * n       # pref_dicts[j] = {gap: prefix_max_value}
    arg_dicts = [None] * n        # arg_dicts[j] = {gap: the h that achieves pref}

    for j in range(n):
        if time.time() > deadline:
            break

        if j == 0:
            best_dicts[j] = {}
            pref_dicts[j] = {}
            arg_dicts[j] = {}
            continue

        # For each i < j, compute candidate
        # gap = primes[j] - primes[i]
        # candidate_len = pref_dicts[i].get(gap, 0) + 1 if pref_dicts[i].get(gap, 0) > 0 else 2
        # Track max candidate per gap value

        best_j_dict = {}
        # We'll accumulate: for each gap, the best (candidate_len, i, parent_gap_at_i)
        # Use a dict: gap -> (max_candidate, best_i, parent_gap_of_best_i)

        gap_best = {}  # gap -> (candidate_len, i, parent_gap_at_i_for_this_gap)

        pj = int(primes[j])
        for i in range(j):
            gap = pj - int(primes[i])
            if gap > max_gap:
                continue

            # Look up pref_dicts[i] for this gap
            pd_i = pref_dicts[i]
            if pd_i is None:
                prev_val = 0
            else:
                prev_val = pd_i.get(gap, 0)

            if prev_val > 0:
                cand = prev_val + 1
                # The parent gap at i for this transition: arg_dicts[i][gap]
                ad_i = arg_dicts[i]
                if ad_i is None:
                    pg_at_i = 0
                else:
                    pg_at_i = ad_i.get(gap, 0)
            else:
                cand = 2
                pg_at_i = 0  # starting fresh, no prior gap

            # Update gap_best for this gap
            if gap not in gap_best:
                gap_best[gap] = (cand, i, pg_at_i)
            else:
                existing = gap_best[gap]
                if cand > existing[0]:
                    gap_best[gap] = (cand, i, pg_at_i)

        # Build best_dicts[j] from gap_best
        best_dicts[j] = {g: (v[0], v[1], v[2]) for g, v in gap_best.items()}

        # Compute prefix max and arg for this j
        # Get sorted gaps
        if not gap_best:
            pref_dicts[j] = {}
            arg_dicts[j] = {}
            continue

        sorted_gaps = sorted(gap_best.keys())
        pref_j = {}
        arg_j = {}
        running_max = 0
        running_arg = 0

        for g in sorted_gaps:
            val = gap_best[g][0]
            if val > running_max:
                running_max = val
                running_arg = g
            pref_j[g] = running_max
            arg_j[g] = running_arg

        pref_dicts[j] = pref_j
        arg_dicts[j] = arg_j

    # Find the best (j, gap)
    max_len = 0
    best_j = -1
    best_gap = -1
    for j in range(n):
        bd = best_dicts[j]
        if bd is None:
            continue
        for g, (length, _, _) in bd.items():
            if length > max_len:
                max_len = length
                best_j = j
                best_gap = g

    # If no valid sequence of length >= 2 found, return the first prime
    if max_len < 2 or best_j < 0:
        return [int(primes[0])]

    # Reconstruct
    result = []
    j = best_j
    gap = best_gap
    while j >= 0:
        result.append(int(primes[j]))
        bd = best_dicts[j]
        if bd is None or gap not in bd:
            break
        _, pi, pg = bd[gap]
        if pi < 0:
            break
        j = pi
        gap = pg
        if gap == 0:
            result.append(int(primes[j]))
            break

    result.reverse()

    # Validate
    if len(result) >= 2:
        gaps_list = [result[i+1] - result[i] for i in range(len(result)-1)]
        valid = all(gaps_list[i+1] >= gaps_list[i] for i in range(len(gaps_list)-1))
        if not valid:
            # Fallback: greedy
            values = []
            for p in range(2, N + 1):
                if is_prime[p]:
                    if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
                        values.append(p)
            return values

    return result

# EVOLVE-BLOCK-END
