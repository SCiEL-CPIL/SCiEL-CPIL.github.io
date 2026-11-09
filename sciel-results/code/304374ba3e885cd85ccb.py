"""Structural change: greedy set-cover growth instead of build-then-prune."""
import time
import numpy as np

N = 18
TIME_LIMIT = 88.0  # safety margin before 90s kill


def _initial_bad_mask():
    """Return the boolean mask of all 2^N inputs that are NOT yet sorted.

    An input (bitstring) is 'sorted' iff all 0-bits precede all 1-bits,
    i.e. it is of the form 0^k 1^{N-k}. There are N+1 such values.
    """
    total = 1 << N
    x = np.arange(total, dtype=np.uint64)
    # sorted iff (x & (x - 1)) == 0  (x is 0 or a power of two)
    is_sorted = (x & (x - 1)) == 0
    return ~is_sorted  # True where input is still bad


def _apply_mask(mask, i, j):
    """Return the mask of inputs that comparator (i,j) fixes (bad -> sorted).

    A comparator fixes input x iff x is bad AND x has bit i = 1, bit j = 0
    (the out-of-order pair).
    """
    bit_i = (mask >> i) & 1
    bit_j = (mask >> j) & 1
    return (bit_i == 1) & (bit_j == 0)


def construct():
    start = time.time()
    total = 1 << N

    bad = _initial_bad_mask()
    if not bad.any():
        return []

    net = []
    bad_count = int(bad.sum())

    # Precompute bit masks for fast extraction
    # We process all pairs each round; N=18 -> 153 pairs, 262144 mask.
    while bad_count > 0:
        if time.time() - start > TIME_LIMIT - 2.0:
            break

        # Evaluate every unordered pair (i,j)
        best_i = -1
        best_j = -1
        best_fixes = -1
        best_mask = None

        for i in range(N):
            for j in range(i + 1, N):
                m = _apply_mask(bad, i, j)
                c = int(m.sum())
                if c > best_fixes:
                    best_fixes = c
                    best_i = i
                    best_j = j
                    best_mask = m

        if best_i < 0 or best_fixes <= 0:
            # No single comparator can reduce bad set -> network incomplete.
            # (Should not happen for a complete network; bail safely.)
            break

        net.append((best_i, best_j))
        bad = bad & ~best_mask
        bad_count -= best_fixes

    # If we ran out of time before completing, fall back to a known valid network
    # only if the grown one is invalid. But we cannot verify the grown one cheaply
    # without the full mask; instead, if incomplete, pad with a valid construction.
    if bad_count > 0:
        # Fallback: append a full odd-even transposition to guarantee validity.
        # (Rare; only on timeout.)
        for phase in range(N):
            if phase % 2 == 0:
                for i in range(0, N - 1, 2):
                    net.append((i, i + 1))
            else:
                for i in range(1, N - 1, 2):
                    net.append((i, i + 1))

    return net
