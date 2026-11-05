"""Program for the suite task; Sidon base with bitset-optimized greedy extension."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n = 512
    max_card = 48

    SIZE = 60001
    OFFSET = 30000

    def has_nonempty_zero_sum(diffs):
        """Check if any non-empty subset of diffs sums to 0."""
        if not diffs:
            return False
        # If any diff is 0, size-1 subset works
        for d in diffs:
            if d == 0:
                return True
        # Build bitset for non-empty subsets
        bits = np.zeros(SIZE, dtype=bool)
        for d in diffs:
            temp = bits.copy()
            if d > 0:
                bits[d:] |= temp[:-d]
            elif d < 0:
                bits[:d] |= temp[-d:]
            idx = OFFSET + d
            if 0 <= idx < SIZE:
                bits[idx] = True
        return bool(bits[OFFSET])

    def is_subset_sum(diffs, target):
        """Check if target is a subset sum of diffs (empty set allowed)."""
        bits = np.zeros(SIZE, dtype=bool)
        bits[OFFSET] = True
        for d in diffs:
            if d == 0:
                continue
            temp = bits.copy()
            if d > 0:
                bits[d:] |= temp[:-d]
            elif d < 0:
                bits[:d] |= temp[-d:]
        idx = OFFSET + target
        if 0 <= idx < SIZE:
            return bool(bits[idx])
        return False

    def can_add(x, S):
        if not S:
            return True

        # Check 1: Is x the mean of any subset of S of size >= 2?
        diffs_x = [s - x for s in S]
        if has_nonempty_zero_sum(diffs_x):
            return False

        # Check 2: For each a in S, check if a is mean of subset of (S U {x})\{a}
        for a in S:
            target = a - x
            diffs_a = [s - a for s in S if s != a]
            if not diffs_a:
                continue
            if target == 0:
                if has_nonempty_zero_sum(diffs_a):
                    return False
            else:
                if is_subset_sum(diffs_a, target):
                    return False

        return True

    # Sidon base: a_i = i*B + i^2
    B = 18
    S = []
    for i in range(1, 20):
        val = i * B + i * i
        if val <= n:
            S.append(val)
        else:
            break

    current = set(S)
    for x in range(1, n + 1):
        if x in current:
            continue
        if can_add(x, S):
            S.append(x)
            current.add(x)
            if len(S) >= max_card:
                break

    return S
# EVOLVE-BLOCK-END
