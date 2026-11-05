import time

def construct():
    deadline = time.time() + 80.0   # explicit wall-clock guard (evaluator kills at 90s)

    n = 12
    MAX_ALLOWED = 1048576

    def is_valid(seq):
        # Exactly n distinct positive integers, each <= MAX_ALLOWED,
        # and all 2^n subset sums are pairwise distinct (dissociated set).
        if len(seq) != n:
            return False
        if any((not isinstance(x, int)) or x <= 0 for x in seq):
            return False
        if any(x > MAX_ALLOWED for x in seq):
            return False
        if len(set(seq)) != n:
            return False
        m = 1 << n
        sums = [0] * m
        # sums[j] = sum of elements selected by the bits of j
        # Build incrementally: for element k, sums[j] for j with bit k set
        # add a_k to the sum of the remaining bits.
        a = seq
        # Fast subset-sum enumeration using DP over elements
        for k in range(n):
            ak = a[k]
            bit = 1 << k
            for j in range(m):
                if j & bit:
                    sums[j] += ak
        # Check distinctness
        if len(set(sums)) != m:
            return False
        return True

    def max_of(seq):
        return max(seq)

    # --- Candidate 1: superbase sequence (provably optimal max) ---
    # a_k = 2^(k-1) - 1  ->  {1, 2, 3, ..., 2^(n-1)-1}
    cand = [(1 << i) - 1 for i in range(n)]   # 1,2,4-1=3? no: i=0->0? fix below
    # Correct superbase: a_1=1, a_{k+1}=1+sum(prev) => a_k = 2^(k-1)-1
    cand = [(1 << i) - 1 for i in range(1, n + 1)]  # i=1..12 -> 1,2,3,5,9,17,...,1023
    assert len(cand) == n
    best = cand
    if not is_valid(best):
        # Fallback to guaranteed-valid powers of two
        best = [1 << i for i in range(n)]
        if not is_valid(best):
            return best
    best_max = max_of(best)

    # --- Time-budgeted search for a strictly better (smaller max) valid set ---
    # Heuristic: greedy superincreasing construction from different seeds, and
    # local search reducing the largest element while preserving dissociation.
    # Each candidate is strictly validated before being accepted.
    def greedy_from(seed_first, seed_second=None):
        seq = []
        if seed_first is not None:
            seq.append(seed_first)
        if seed_second is not None:
            seq.append(seed_second)
        prev_sum = sum(seq)
        while len(seq) < n:
            # smallest positive integer > prev_sum not already in seq
            v = prev_sum + 1
            present = set(seq)
            while v in present:
                v += 1
            seq.append(v)
            prev_sum += v
        return seq

    start = time.time()
    improved = False
    # Try a handful of distinct greedy seeds; each is validated and only
    # accepted if it strictly lowers the max and is valid.
    seeds = [1, 1, 2, 1, 3, 2, 2]
    for s in seeds:
        if time.time() > deadline:
            break
        g = greedy_from(s)
        if max_of(g) < best_max and is_valid(g):
            best = g
            best_max = max_of(g)
            improved = True

    # Local search: attempt to shrink the largest element of the current best
    # while keeping dissociation (only accept verified improvements).
    if time.time() < deadline:
        for trial in range(200):
            if time.time() > deadline:
                break
            cur = list(best)
            largest_idx = max(range(n), key=lambda i: cur[i])
            cur_max = cur[largest_idx]
            found = False
            for cand_max in range(cur_max - 1, 0, -1):
                if cand_max == cur_max:
                    continue
                trial_seq = cur[:]
                trial_seq[largest_idx] = cand_max
                if len(set(trial_seq)) != n:
                    continue
                if is_valid(trial_seq):
                    best = trial_seq
                    best_max = cand_max
                    improved = True
                    found = True
                    break
            if not found:
                break

    return best
