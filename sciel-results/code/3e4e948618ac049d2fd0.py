"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import time

    deadline = time.time() + 75.0  # 75s deadline, 15s safety margin
    best_deficit = Fraction(1, 1)  # initial: empty set, deficit = 1
    best_set = []

    denominators = list(range(2, 49))  # 47 denominators
    n = len(denominators)

    # Use meet-in-the-middle approach
    # Split into two halves: first 23 and last 24
    mid = 23
    left_denoms = denominators[:mid]
    right_denoms = denominators[mid:]
    n_left = len(left_denoms)
    n_right = len(right_denoms)

    # For exact arithmetic, use a common denominator (LCM)
    # LCM of 2..48 is huge, but we can use it for integer arithmetic
    from math import gcd
    from functools import reduce

    lcm_val = reduce(lambda a, b: a * b // gcd(a, b), denominators)

    # Precompute numerator for each denominator
    left_nums = [lcm_val // d for d in left_denoms]
    right_nums = [lcm_val // d for d in right_denoms]

    # Generate all subset sums for left half
    left_sums = {}  # sum -> list of indices (bitmask representation)
    for mask in range(1 << n_left):
        s = 0
        for i in range(n_left):
            if mask & (1 << i):
                s += left_nums[i]
        if s < lcm_val:  # strict underpacking
            if s > 0:  # nonempty
                if s not in left_sums:
                    left_sums[s] = []
                left_sums[s].append(mask)

    # Generate all subset sums for right half
    right_sums = {}
    for mask in range(1 << n_right):
        s = 0
        for i in range(n_right):
            if mask & (1 << i):
                s += right_nums[i]
        if s < lcm_val:
            if s > 0:
                if s not in right_sums:
                    right_sums[s] = []
                right_sums[s].append(mask)

    # Sort keys for binary search
    left_keys = sorted(left_sums.keys())
    right_keys = sorted(right_sums.keys())

    import bisect

    # For each right sum, find the best left sum such that left + right < lcm_val
    best_total = 0  # in units of lcm_val
    best_left_mask = -1
    best_right_mask = -1

    # To speed up, we can iterate over right sums and binary search left sums
    for r_sum in right_keys:
        # Need left_sum < lcm_val - r_sum
        max_left = lcm_val - r_sum - 1  # strict: left + right < lcm_val
        if max_left < 0:
            continue
        # Find largest left_sum <= max_left
        idx = bisect.bisect_right(left_keys, max_left) - 1
        if idx >= 0:
            l_sum = left_keys[idx]
            total = l_sum + r_sum
            if total > best_total:
                best_total = total
                best_left_mask = left_sums[l_sum][0]
                best_right_mask = right_sums[r_sum][0]

        # Check time
        if time.time() > deadline:
            break

    # If no combination found, try just left or just right
    if best_total == 0:
        if left_keys:
            best_total = left_keys[-1]
            best_left_mask = left_sums[best_total][0]
            best_right_mask = -1
        elif right_keys:
            best_total = right_keys[-1]
            best_left_mask = -1
            best_right_mask = right_sums[best_total][0]

    # Convert masks to indices
    selected = []
    if best_left_mask >= 0:
        for i in range(n_left):
            if best_left_mask & (1 << i):
                selected.append(i)
    if best_right_mask >= 0:
        for i in range(n_right):
            if best_right_mask & (1 << i):
                selected.append(mid + i)

    if not selected:
        # Fallback: greedy
        selected = []
        total = Fraction()
        for i, d in enumerate(denominators):
            if total + Fraction(1, d) < 1:
                selected.append(i)
                total += Fraction(1, d)

    return selected
# EVOLVE-BLOCK-END
