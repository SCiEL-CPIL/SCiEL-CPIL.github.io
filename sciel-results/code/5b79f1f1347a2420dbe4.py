import time
import numpy as np
from bisect import bisect_right

def construct():
    """
    Finds the longest subsequence of primes <= 10000 with non-decreasing gaps.
    Uses Pareto-front compression: for each prime, store only the minimal
    (gap, length) pairs that form the upper envelope. This reduces memory
    from O(N*G) to O(N*L) where L is the Pareto frontier size.
    """
    deadline = time.time() + 80.0
    LIMIT = 10000

    # Generate primes
    sieve = np.ones(LIMIT + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(LIMIT**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    primes = np.array([i for i, is_p in enumerate(sieve) if is_p], dtype=np.int32)
    N = len(primes)

    # For each prime i, maintain a Pareto frontier of (gap, length) pairs.
    # A pair (g, l) is in the frontier if no other pair (g', l') has g' <= g and l' >= l
    # (with at least one strict inequality).
    # We store these as two parallel lists: gaps[i] (sorted ascending), lens[i] (corresponding).
    # Since gaps are sorted ascending and we keep only Pareto-optimal points,
    # lens[i] will be strictly increasing.
    #
    # Additionally, we track parent pointers for reconstruction.
    # parent[i][k] = (j, h) where j is the predecessor prime index and h is the
    # frontier index in prime j that was used to create state k of prime i.

    front_gaps = [None] * N
    front_lens = [None] * N
    front_parent = [None] * N  # list of (j, h) tuples

    # Prime 0: single element, gap = 0, length = 1
    front_gaps[0] = [0]
    front_lens[0] = [1]
    front_parent[0] = [(-1, -1)]

    best_len = 1
    best_i = 0
    best_k = 0

    for i in range(1, N):
        if time.time() > deadline:
            break

        pi = primes[i]

        # For each previous prime j < i, compute gap = pi - primes[j].
        # The best length we can extend from j with last_gap <= gap is:
        #   best_l = max lens[j][h] for all h where gaps[j][h] <= gap
        # Since gaps[j] is sorted, we use bisect_right to find the rightmost
        # index where gaps[j][h] <= gap, then take lens[j][that_index].
        #
        # We collect all (gap, best_l+1) candidates, then compute the Pareto frontier.

        candidates = []  # list of (gap, length, j, h)

        for j in range(i):
            gap = int(pi - primes[j])
            fj_gaps = front_gaps[j]
            fj_lens = front_lens[j]

            # Find rightmost index in fj_gaps where value <= gap
            idx = bisect_right(fj_gaps, gap)
            if idx > 0:
                best_l = fj_lens[idx - 1] + 1
                h = idx - 1  # the frontier index used
                candidates.append((gap, best_l, j, h))
            else:
                # No valid predecessor gap <= gap; can only start a new sequence
                # (length 1, but that's already covered by the base case)
                pass

        if not candidates:
            front_gaps[i] = [0]
            front_lens[i] = [1]
            front_parent[i] = [(-1, -1)]
            continue

        # Also include the "start new sequence" option: gap=0, length=1
        # But actually, if we're at prime i, we can always have a sequence of length 1.
        # The gap for a single-element sequence is 0 (no gap).
        # However, for the purpose of extending further, a single-element sequence
        # has "last gap" = 0. So we include (0, 1) as a candidate.
        # But wait: if we have a candidate with gap=0 and length=1, it's dominated
        # by any candidate with gap>=0 and length>=1. Since all candidates have
        # length >= 2 (they extend from some j), the (0,1) is always dominated
        # unless there are no other candidates. Let's handle this:
        # Actually, we should always include the option of starting fresh at i.
        # Let's add it and let Pareto filtering handle it.
        candidates.append((0, 1, -1, -1))

        # Now compute Pareto frontier from candidates.
        # Sort by gap ascending, then by length descending (so longer sequences
        # come first for same gap, and we can easily find the frontier).
        candidates.sort(key=lambda x: (x[0], -x[1]))

        # Build Pareto frontier: iterate through sorted candidates, keeping only
        # those with length > all previously kept lengths.
        fg = []
        fl = []
        fp = []
        max_l_so_far = 0

        for gap, length, j, h in candidates:
            if length > max_l_so_far:
                fg.append(gap)
                fl.append(length)
                fp.append((j, h))
                max_l_so_far = length

        front_gaps[i] = fg
        front_lens[i] = fl
        front_parent[i] = fp

        # Update global best
        if max_l_so_far > best_len:
            best_len = max_l_so_far
            best_i = i
            best_k = len(fg) - 1  # the last (highest length) entry in the frontier

    # Reconstruct sequence
    seq = []
    i = best_i
    k = best_k
    while i >= 0:
        seq.append(int(primes[i]))
        j, h = front_parent[i][k]
        if j == -1:
            break
        i = j
        k = h
    seq.reverse()

    return seq
