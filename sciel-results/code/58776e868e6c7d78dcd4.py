"""Combined: length-indexed DP for longest nondecreasing-gap prime subsequence."""
import time
import bisect

_DEADLINE = time.time() + 80.0


def construct():
    deadline = _DEADLINE
    LIMIT = 10000

    # --- Sieve primes up to LIMIT ---
    sieve = bytearray(b'\x01') * (LIMIT + 1)
    sieve[0:2] = b'\x00\x00'
    for x in range(2, int(LIMIT ** 0.5) + 1):
        if sieve[x]:
            sieve[x * x::x] = b'\x00' * len(sieve[x * x::x])
    primes = [i for i in range(2, LIMIT + 1) if sieve[i]]
    n = len(primes)

    # --- Length-indexed DP ---
    # dp[L] = list of (prime_value, min_last_gap) for sequences of length L
    # min_last_gap = smallest gap used in the last step among all valid seqs of length L ending at that prime
    INF = 10**9
    dp = [[] for _ in range(n + 1)]  # dp[0] unused
    dp[1] = [(p, 0) for p in primes]  # length-1: no gap, use 0 as sentinel

    pred = [{} for _ in range(n + 1)]  # pred[L][(p, gap)] = (prev_p, prev_gap)
    max_L = 1

    for L in range(2, n + 1):
        if time.time() > deadline:
            break
        new_dp = []
        # For each prime as the last element, find the best (min last gap) predecessor
        # We want: for prime q, find prime p < q such that (q - p) >= gap_p
        # and gap_p is minimized (to leave room for future extensions)
        #
        # Efficient approach: iterate primes in order, maintain a structure of
        # (gap, prime) from dp[L-1] to quickly find valid predecessors.
        #
        # For each candidate last prime q:
        #   Need p from dp[L-1] with p < q and gap_p <= (q - p)
        #   i.e., gap_p + p <= q
        #   Among valid p, we want the one minimizing (q - p) = the last gap
        #   which means maximizing p (closest valid predecessor)
        #
        # So: from dp[L-1], we need the largest p such that (p + gap_p) <= q.
        # Precompute: for dp[L-1], sort by (p + gap_p), then for each q we binary search.

        prev = dp[L - 1]
        if not prev:
            break

        # Build sorted structure: key = p + gap_p, value = (p, gap_p)
        # Sort by key, and for same key keep the largest p (to maximize gap = q - p)
        entries = sorted(prev, key=lambda x: (x[0] + x[1], -x[0]))
        keys = [e[0] + e[1] for e in entries]  # p + gap_p
        # For binary search: find rightmost entry with key <= q
        # But we also want the largest p among those, which is the rightmost in sorted order
        # (since we sorted by key then -p, the rightmost with key<=q has the largest p)

        for q in primes:
            # Find rightmost index where key <= q
            idx = bisect.bisect_right(keys, q) - 1
            if idx < 0:
                continue
            p, gap_p = entries[idx]
            if p >= q:
                continue
            new_gap = q - p
            # This is valid: gap_p <= new_gap (since p + gap_p <= q means gap_p <= q - p)
            # Store: for (q, new_gap), predecessor is (p, gap_p)
            # But we want min new_gap per q... actually we want to store the best option per q
            # The largest p gives the smallest new_gap, which is best for future extensions
            if not new_dp or new_dp[-1][0] != q:
                new_dp.append((q, new_gap))
                pred[L][(q, new_gap)] = (p, gap_p)
            # Since we iterate q in increasing order and entries are fixed,
            # the first valid p found (largest p with p+gap_p <= q) gives min new_gap
            # Actually bisect_right gives the rightmost, which IS the largest p. Good.

        if not new_dp:
            break
        dp[L] = new_dp
        max_L = L

    # --- Reconstruct the sequence ---
    if max_L < 2:
        # Fallback: greedy
        from math import isqrt
        vals = []
        for p in range(2, 10001):
            if all(p % d for d in range(2, isqrt(p) + 1)):
                if len(vals) < 2 or p - vals[-1] >= vals[-1] - vals[-2]:
                    vals.append(p)
        return vals

    # Start from the last element of dp[max_L] (any is fine; pick last for determinism)
    q, gap = dp[max_L][-1]
    seq = [q]
    L = max_L
    while L > 1:
        prev_p, prev_gap = pred[L][(q, gap)]
        seq.append(prev_p)
        q, gap = prev_p, prev_gap
        L -= 1
    seq.reverse()

    # --- Validate and return ---
    if len(seq) < 2:
        from math import isqrt
        vals = []
        for p in range(2, 10001):
            if all(p % d for d in range(2, isqrt(p) + 1)):
                if len(vals) < 2 or p - vals[-1] >= vals[-1] - vals[-2]:
                    vals.append(p)
        return vals

    return seq
