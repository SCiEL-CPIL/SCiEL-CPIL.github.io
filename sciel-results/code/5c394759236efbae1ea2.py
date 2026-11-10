# EVOLVE-BLOCK-START
import time
import numpy as np


def _base_digits(q, n, x):
    # x in [0, q**n) -> list of n digits in 0..q-1 (big-endian)
    digs = [0] * n
    for i in range(n - 1, -1, -1):
        digs[i] = x % q
        x //= q
    return digs


def _all_neighbors(q, n, radius):
    # Return list of tuples: each tuple is a list of (position, delta) with len <= radius
    # For radius 1: (0,0),(0,1),(0,2),...,(n-1,0),(n-1,1),(n-1,2)
    if radius == 0:
        return [()]
    if radius == 1:
        res = []
        for pos in range(n):
            for d in range(q):
                res.append(((pos, d),))
        return res
    # general radius (fallback)
    res = []
    import itertools
    for r in range(1, radius + 1):
        for pos in itertools.combinations(range(n), r):
            for deltas in itertools.product(range(q), repeat=r):
                res.append(tuple(zip(pos, deltas)))
    return res


def _apply(c, neighbors):
    # apply neighbor offsets to codeword c (list), return list
    out = c[:]
    for pos, d in neighbors:
        out[pos] = d
    return out


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 105.0  # safety margin under 120s

    N = q ** n
    neighbors = _all_neighbors(q, n, radius)
    n_nbr = len(neighbors)

    # Precompute for each codeword index its covered indices (including itself)
    # covered_by[c] = array of covered word indices
    covered_by = np.empty(N, dtype=object)
    for c in range(N):
        base = _base_digits(q, n, c)
        cov = []
        for nb in neighbors:
            w = _apply(base, nb)
            idx = 0
            for d in w:
                idx = idx * q + d
            cov.append(idx)
        covered_by[c] = np.array(cov, dtype=np.int64)

    # Initial greedy solution
    covered = np.zeros(N, dtype=bool)
    code = []
    # Precompute order? Just greedy.
    while not covered.all():
        if time.time() > deadline - 5.0:
            break
        # find codeword covering most uncovered
        # For speed, only consider a subset if N is large
        # Compute counts
        best = -1
        best_count = -1
        # Vectorized-ish: for each candidate c, count uncovered in covered_by[c]
        # We'll loop but use numpy for the count
        for c in range(N):
            cov = covered_by[c]
            cnt = np.count_nonzero(~covered[cov])
            if cnt > best_count:
                best_count = cnt
                best = c
                if best_count == n_nbr:
                    # can't do better
                    break
        if best < 0:
            break
        code.append(best)
        covered[covered_by[best]] = True

    # Local search improvement
    def uncovered_count():
        return int(np.count_nonzero(~covered))

    def recompute_covered(code_list):
        cov = np.zeros(N, dtype=bool)
        for c in code_list:
            cov[covered_by[c]] = True
        return cov

    # Build count of how many codewords cover each word (for removal feasibility)
    def build_counts(code_list):
        counts = np.zeros(N, dtype=np.int64)
        for c in code_list:
            counts[covered_by[c]] += 1
        return counts

    counts = build_counts(code)

    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        # Try removing each codeword if it leaves few uniquely covered words
        for i in range(len(code)):
            if time.time() > deadline - 5.0:
                break
            c = code[i]
            # uniquely covered words by c: covered only by c
            # counts[cov] == 1 for words only c covers
            cov = covered_by[c]
            unique_mask = (counts[cov] == 1)
            n_unique = int(np.count_nonzero(unique_mask))
            if n_unique == 0:
                # safe to remove, no coverage loss
                new_code = code[:i] + code[i + 1:]
                code = new_code
                counts = build_counts(code)
                covered = recompute_covered(code)
                improved = True
                continue
            # If removing loses few words, try to re-add one codeword covering them
            if n_unique <= 1:
                # candidate: remove c, then find best codeword to cover the unique words
                # unique words indices
                uniq_idx = cov[unique_mask]
                # Try each possible new codeword; pick one that covers most of uniq_idx
                # and doesn't break others (it can't break others, only add)
                best_new = -1
                best_gain = -1
                for nc in range(N):
                    ncov = covered_by[nc]
                    gain = int(np.isin(uniq_idx, ncov).sum())
                    if gain > best_gain:
                        best_gain = gain
                        best_new = nc
                        if best_gain == n_unique:
                            break
                if best_new >= 0 and best_gain >= n_unique and best_new not in code:
                    # replace
                    new_code = code[:i] + code[i + 1:] + [best_new]
                    code = new_code
                    counts = build_counts(code)
                    covered = recompute_covered(code)
                    improved = True

    # Final safety: ensure full coverage; if not, greedily add
    if not covered.all():
        while not covered.all() and time.time() < deadline - 2.0:
            best = -1
            best_count = -1
            for c in range(N):
                if c in code:
                    continue
                cov = covered_by[c]
                cnt = np.count_nonzero(~covered[cov])
                if cnt > best_count:
                    best_count = cnt
                    best = c
            if best < 0:
                break
            code.append(best)
            covered[covered_by[best]] = True

    # Convert to list of digit lists
    result = [_base_digits(q, n, c) for c in code]
    # Ensure distinct (they are, since c distinct)
    return result
# EVOLVE-BLOCK-END
