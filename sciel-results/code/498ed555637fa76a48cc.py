def construct():
    from fractions import Fraction
    import time
    import bisect

    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill

    denominators = list(range(2, 49))
    n = len(denominators)  # 47 items

    # Split into two halves
    mid = n // 2  # 23
    left_items = list(range(mid))      # indices 0..22 (denoms 2..24)
    right_items = list(range(mid, n))  # indices 23..46 (denoms 25..48)

    # Generate all valid subsets (sum < 1) for the left half
    left_subsets = []  # list of (sum, bitmask)
    for mask in range(1 << len(left_items)):
        s = Fraction(0)
        for i in range(len(left_items)):
            if mask & (1 << i):
                s += Fraction(1, denominators[left_items[i]])
        if s < 1:
            left_subsets.append((s, mask))

    # Generate all valid subsets (sum < 1) for the right half
    right_subsets = []  # list of (sum, bitmask)
    for mask in range(1 << len(right_items)):
        s = Fraction(0)
        for i in range(len(right_items)):
            if mask & (1 << i):
                s += Fraction(1, denominators[right_items[i]])
        if s < 1:
            right_subsets.append((s, mask))

    # Sort right_subsets by sum for binary search
    right_subsets.sort(key=lambda x: x[0])
    right_sums = [s for s, _ in right_subsets]

    best_sum = Fraction(0)
    best_left_mask = 0
    best_right_mask = 0

    # For each left subset, find the best right subset such that total < 1
    for left_sum, left_mask in left_subsets:
        # We need right_sum < 1 - left_sum
        target = 1 - left_sum
        # Find the largest right_sum < target
        # bisect_left gives first index where right_sums[i] >= target
        # So we want index - 1 for the largest < target
        idx = bisect.bisect_left(right_sums, target)
        if idx > 0:
            candidate_sum = left_sum + right_sums[idx - 1]
            if candidate_sum > best_sum:
                best_sum = candidate_sum
                best_left_mask = left_mask
                best_right_mask = right_subsets[idx - 1][1]

    # Check time
    if time.time() > deadline:
        # Return best found so far
        indices = []
        for i in range(len(left_items)):
            if best_left_mask & (1 << i):
                indices.append(left_items[i])
        for i in range(len(right_items)):
            if best_right_mask & (1 << i):
                indices.append(right_items[i])
        if not indices:
            indices = [0]
        return indices

    # Convert masks to indices
    indices = []
    for i in range(len(left_items)):
        if best_left_mask & (1 << i):
            indices.append(left_items[i])
    for i in range(len(right_items)):
        if best_right_mask & (1 << i):
            indices.append(right_items[i])

    if not indices:
        indices = [0]

    return indices
