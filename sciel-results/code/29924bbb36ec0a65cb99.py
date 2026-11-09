import time

def construct():
    start_time = time.time()
    timeout = 55.0

    # Generate 5-smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        while y % 2 == 0: y //= 2
        while y % 3 == 0: y //= 3
        while y % 5 == 0: y //= 5
        if y == 1:
            smooth.append(x)
    smooth.sort()
    n = len(smooth)

    # Precompute divisibility
    divs = [[False]*n for _ in range(n)]
    for i in range(n):
        vi = smooth[i]
        for j in range(i+1, n):
            if smooth[j] % vi == 0:
                divs[i][j] = True
                divs[j][i] = True

    # For each target, find minimum antichain representation
    # Use backtracking with good pruning
    best = [None] * 129  # best[target] = (size, tuple_of_indices)

    # Sort smooth numbers in descending order for better pruning
    # But we need to maintain antichain constraint
    # Let's try a different approach: iterate by size

    # For small targets, try size 1, then 2, then 3, etc.
    for target in range(1, 129):
        if time.time() - start_time > timeout:
            break
        # Try increasing sizes
        for size in range(1, 31):
            found = _find_antichain(target, size, smooth, divs, n, start_time, timeout)
            if found is not None:
                best[target] = (size, found)
                break
        if best[target] is None:
            # Fallback: find any valid representation
            found = _find_any(target, smooth, divs, n, start_time, timeout)
            if found:
                best[target] = (len(found), tuple(found))

    # Build final answers
    final = []
    for t in range(1, 129):
        if best[t] is not None:
            size, indices = best[t]
            final.append([smooth[i] for i in indices])
        else:
            # Should not happen, but fallback
            final.append([t] if t in smooth else [1, t-1] if t-1 > 0 else [t])

    return final

def _find_antichain(target, size, smooth, divs, n, start_time, timeout):
    """Find an antichain of exactly 'size' smooth numbers summing to target."""
    if time.time() - start_time > timeout:
        return None

    # Use backtracking with pruning
    # We want to pick 'size' distinct indices with no divisibility, summing to target
    # Sort indices in descending order of value for better pruning

    # Precompute: for each index, which other indices are compatible
    # compatible[i] = set of j > i such that not divs[i][j]
    # But this depends on the specific subset, so we check on the fly

    # Strategy: try combinations in a smart order
    # Start with larger numbers to reduce search space

    # Get indices sorted by value descending
    idx_order = list(range(n-1, -1, -1))

    # Pruning: if we need 'size' more numbers and the smallest possible sum exceeds target, prune
    # Also if the largest possible sum of 'size' remaining is less than target, prune

    def backtrack(start_idx, current_sum, current_size, current_set):
        if time.time() - start_time > timeout:
            return None

        if current_size == size:
            if current_sum == target:
                return list(current_set)
            return None

        if current_sum > target:
            return None

        remaining_needed = size - current_size
        if remaining_needed <= 0:
            return None

        # Pruning: minimum possible sum of remaining_needed numbers from remaining indices
        # The smallest remaining numbers
        min_possible = 0
        count = 0
        for i in range(start_idx, n):
            if count >= remaining_needed:
                break
            # Check if this index is compatible with all in current_set
            compatible = True
            for j in current_set:
                if divs[i][j]:
                    compatible = False
                    break
            if compatible:
                min_possible += smooth[i]
                count += 1
        if count < remaining_needed:
            return None
        if current_sum + min_possible > target:
            return None

        # Maximum possible sum
        max_possible = 0
        count = 0
        for i in range(n-1, start_idx-1, -1):
            if count >= remaining_needed:
                break
            compatible = True
            for j in current_set:
                if divs[i][j]:
                    compatible = False
                    break
            if compatible:
                max_possible += smooth[i]
                count += 1
        if count < remaining_needed:
            return None
        if current_sum + max_possible < target:
            return None

        for i in range(start_idx, n):
            val = smooth[i]
            if val > target - current_sum:
                break

            # Check compatibility
            compatible = True
            for j in current_set:
                if divs[i][j]:
                    compatible = False
                    break
            if not compatible:
                continue

            current_set.append(i)
            result = backtrack(i+1, current_sum + val, current_size + 1, current_set)
            current_set.pop()
            if result is not None:
                return result

        return None

    return backtrack(0, 0, 0, [])

def _find_any(target, smooth, divs, n, start_time, timeout):
    """Find any valid antichain representation for target."""
    if time.time() - start_time > timeout:
        return None

    def backtrack(start_idx, current_sum, current_set):
        if time.time() - start_time > timeout:
            return None

        if current_sum == target:
            return list(current_set)

        if current_sum > target:
            return None

        for i in range(start_idx, n):
            val = smooth[i]
            if val > target - current_sum:
                break

            compatible = True
            for j in current_set:
                if divs[i][j]:
                    compatible = False
                    break
            if not compatible:
                continue

            current_set.append(i)
            result = backtrack(i+1, current_sum + val, current_set)
            current_set.pop()
            if result is not None:
                return result

        return None

    result = backtrack(0, 0, [])
    if result:
        return [smooth[i] for i in result]
    return None
