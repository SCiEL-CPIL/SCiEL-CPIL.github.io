def construct():
    from fractions import Fraction
    import time
    import bisect

    deadline = time.time() + 75  # Safety margin before 90s limit

    denominators = list(range(2, 49))
    n = len(denominators)
    fractions = [Fraction(1, d) for d in denominators]

    # Split into two halves for meet-in-the-middle
    mid = 23
    left_fracs = fractions[:mid]
    right_fracs = fractions[mid:]
    left_n = mid
    right_n = n - mid

    # Generate all subset sums for the left half
    # Store as (sum, bitmask) pairs
    left_sums = [(Fraction(0), 0)]
    for i in range(left_n):
        f = left_fracs[i]
        current_len = len(left_sums)
        for j in range(current_len):
            s, mask = left_sums[j]
            new_s = s + f
            if new_s < 1:
                left_sums.append((new_s, mask | (1 << i)))

    # Generate all subset sums for the right half
    right_sums = [(Fraction(0), 0)]
    for i in range(right_n):
        f = right_fracs[i]
        current_len = len(right_sums)
        for j in range(current_len):
            s, mask = right_sums[j]
            new_s = s + f
            if new_s < 1:
                right_sums.append((new_s, mask | (1 << i)))

    # Sort right sums by value for binary search
    right_sums.sort(key=lambda x: x[0])
    right_values = [s for s, _ in right_sums]

    # For each left sum, find the best right sum such that total < 1
    best_sum = Fraction(0)
    best_left_mask = 0
    best_right_mask = 0

    for ls, lmask in left_sums:
        # Find largest rs such that ls + rs < 1
        # i.e., rs < 1 - ls
        target = 1 - ls
        # Binary search for rightmost value < target
        # bisect_left gives first index >= target, so we want index before that
        idx = bisect.bisect_left(right_values, target)
        if idx > 0:
            rs, rmask = right_sums[idx - 1]
            total = ls + rs
            if total > best_sum:
                best_sum = total
                best_left_mask = lmask
                best_right_mask = rmask

    # Reconstruct indices from bitmasks
    best_indices = []
    for i in range(left_n):
        if best_left_mask & (1 << i):
            best_indices.append(i)
    for i in range(right_n):
        if best_right_mask & (1 << i):
            best_indices.append(mid + i)

    # Local search refinement
    if best_indices and time.time() < deadline - 10:
        curr_set = set(best_indices)
        curr_sum = best_sum
        curr_indices = list(best_indices)

        improved = True
        while improved and time.time() < deadline - 5:
            improved = False

            # Try adding elements
            for i in range(n):
                if i not in curr_set:
                    if curr_sum + fractions[i] < 1:
                        new_sum = curr_sum + fractions[i]
                        if new_sum > best_sum:
                            curr_indices.append(i)
                            curr_set.add(i)
                            curr_sum = new_sum
                            best_sum = new_sum
                            best_indices = curr_indices
                            improved = True
                            break

            if improved:
                continue

            # Try swaps: remove i, add j where the swap improves the sum
            for i in curr_indices[:]:
                for j in range(n):
                    if j in curr_set:
                        continue
                    new_sum = curr_sum - fractions[i] + fractions[j]
                    if new_sum < 1 and new_sum > best_sum:
                        curr_indices = [x for x in curr_indices if x != i] + [j]
                        curr_set = set(curr_indices)
                        curr_sum = new_sum
                        best_sum = new_sum
                        best_indices = curr_indices
                        improved = True
                        break
                if improved:
                    break

    return best_indices
