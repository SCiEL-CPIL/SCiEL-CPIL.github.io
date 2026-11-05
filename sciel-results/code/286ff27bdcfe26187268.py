"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # safety margin before 90s kill

    denominators = list(range(2, 49))
    n = len(denominators)  # 47

    # Compute LCM of all denominators to use integer weights
    lcm_val = 1
    for d in denominators:
        lcm_val = lcm_val // gcd(lcm_val, d) * d

    # Integer weights: w[i] = LCM // d[i]
    weights = [lcm_val // d for d in denominators]
    target = lcm_val  # we want sum of weights < target

    # Meet-in-the-middle: split into two halves
    mid = n // 2  # 23
    left_indices = list(range(mid))
    right_indices = list(range(mid, n))
    left_weights = weights[:mid]
    right_weights = weights[mid:]

    # Generate all subset sums for the left half (2^23 = 8,388,608)
    # Store (sum, bitmask) for each subset
    left_sums = []
    left_count = 1 << mid
    for mask in range(left_count):
        s = 0
        m = mask
        idx = 0
        while m:
            if m & 1:
                s += left_weights[idx]
            idx += 1
            m >>= 1
        left_sums.append((s, mask))
        if time.time() > deadline:
            break

    left_sums.sort(key=lambda x: x[0])

    # Generate all subset sums for the right half (2^24 = 16,777,216)
    # For each right subset sum rs, we need the largest left sum ls such that ls + rs < target
    # i.e., ls < target - rs, so we binary search for the largest ls < (target - rs)
    right_count = 1 << (n - mid)

    best_sum = 0  # best total sum found (integer, must be < target)
    best_mask = 0

    for rmask in range(right_count):
        rs = 0
        m = rmask
        idx = 0
        while m:
            if m & 1:
                rs += right_weights[idx]
            idx += 1
            m >>= 1

        if rs >= target:
            continue

        # We need ls < target - rs
        limit = target - rs  # ls must be < limit, i.e., ls <= limit - 1
        # Binary search in left_sums for the largest sum < limit
        lo, hi = 0, len(left_sums) - 1
        best_ls = -1
        best_lmask = 0
        if left_sums:
            if left_sums[0][0] >= limit:
                # No valid left sum
                pass
            else:
                # Find rightmost index where sum < limit
                lo, hi = 0, len(left_sums)
                while lo < hi:
                    mid_i = (lo + hi) // 2
                    if left_sums[mid_i][0] < limit:
                        lo = mid_i + 1
                    else:
                        hi = mid_i
                # lo is the first index where sum >= limit, so lo-1 is the last valid
                if lo > 0:
                    best_ls = left_sums[lo - 1][0]
                    best_lmask = left_sums[lo - 1][1]

        if best_ls >= 0:
            total = best_ls + rs
            if total > best_sum:
                best_sum = total
                best_mask = (best_lmask << (n - mid)) | rmask
                if total == target - 1:
                    # Can't do better than target - 1
                    break

        if time.time() > deadline:
            break

    # Convert mask to indices
    selected = []
    for i in range(n):
        if best_mask & (1 << i):
            selected.append(i)

    # If no valid selection found (shouldn't happen since individual fractions < 1),
    # fall back to greedy
    if not selected:
        selected = [0]  # just 1/2

    # Verify the result is strictly below 1
    total_frac = Fraction(0)
    for i in selected:
        total_frac += Fraction(1, denominators[i])
    if total_frac >= 1:
        # Should not happen, but fallback
        selected = [0]

    return selected
# EVOLVE-BLOCK-END
