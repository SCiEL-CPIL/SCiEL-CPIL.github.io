import time
from fractions import Fraction

def construct():
    start = time.time()
    deadline = start + 80  # safety margin before 90s kill

    denominators = list(range(2, 49))
    n = len(denominators)  # 47

    # LCM of 2..48
    LCM = 5342931457063200
    vals = [LCM // d for d in denominators]

    # Split into two halves
    mid = n // 2  # 23
    left_vals = vals[:mid]
    right_vals = vals[mid:]
    n_left = len(left_vals)
    n_right = len(right_vals)

    # Enumerate all subsets of the left half
    # For each subset, record (sum, mask)
    left_entries = []
    for mask in range(1 << n_left):
        s = 0
        m = mask
        i = 0
        while m:
            if m & 1:
                s += left_vals[i]
            m >>= 1
            i += 1
        left_entries.append((s, mask))
    left_entries.sort()

    # For each subset of the right half, find the best left subset
    # such that total < LCM (i.e., deficit > 0) and total is maximized
    # total = right_sum + left_sum < LCM
    # So left_sum < LCM - right_sum
    # We want to maximize left_sum subject to left_sum < LCM - right_sum
    # i.e., find the largest left_sum < (LCM - right_sum)

    import bisect

    left_sums = [e[0] for e in left_entries]
    left_masks = [e[1] for e in left_entries]

    best_deficit = None
    best_mask = None

    for rmask in range(1 << n_right):
        rs = 0
        m = rmask
        i = 0
        while m:
            if m & 1:
                rs += right_vals[i]
            m >>= 1
            i += 1

        if rs >= LCM:
            continue  # right half alone already >= 1

        # We need left_sum < LCM - rs
        # Find the largest left_sum < (LCM - rs)
        target = LCM - rs  # left_sum must be < target
        # Find rightmost index where left_sums[idx] < target
        idx = bisect.bisect_left(left_sums, target) - 1
        if idx < 0:
            continue

        ls = left_sums[idx]
        total = rs + ls
        deficit = LCM - total
        if deficit <= 0:
            continue

        # Ensure the combined subset is nonempty
        lmask = left_masks[idx]
        if lmask == 0 and rmask == 0:
            continue

        if best_deficit is None or deficit < best_deficit:
            best_deficit = deficit
            best_mask = (lmask, rmask)

        if time.time() > deadline - 5:
            break

    if best_deficit is None or best_mask is None:
        # Fallback: single largest fraction
        return [0]

    lmask, rmask = best_mask
    result = []
    for i in range(n_left):
        if lmask & (1 << i):
            result.append(i)
    for i in range(n_right):
        if rmask & (1 << i):
            result.append(mid + i)

    if not result:
        result = [0]

    # Verify
    total_frac = sum(Fraction(1, denominators[i]) for i in result)
    if total_frac >= 1:
        # Shouldn't happen, but fallback
        return [0]

    return result
