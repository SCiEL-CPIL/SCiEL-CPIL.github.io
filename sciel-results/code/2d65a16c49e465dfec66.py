"""
B_3[1] set construction: maximize cardinality of distinct positive integers <= 4096
such that all unordered triple sums (allowing repetitions) are unique.

Strategy:
1. Fast collision checking using numpy vectorization.
2. Multiple greedy strategies with different starting patterns.
3. Local improvement via swap-based search.
4. Deadline guard to ensure completion within 90s.
"""
import time
import random
import numpy as np


def _all_triple_sums(lst):
    """Compute all unordered triple sums for a sorted list."""
    n = len(lst)
    sums = []
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                sums.append(lst[i] + lst[j] + lst[k])
    return sums


def _is_valid_triples(lst):
    """Check if all unordered triple sums are distinct."""
    if not lst:
        return True
    sums = _all_triple_sums(lst)
    return len(sums) == len(set(sums))


def _check_collision_fast(existing_arr, existing_sums_set, x):
    """
    Fast collision check using numpy vectorization.
    Returns True if adding x would create a collision.
    """
    n = len(existing_arr)
    if n == 0:
        return False

    # Check 3*x
    if 3 * x in existing_sums_set:
        return True

    # Check 2*x + existing[i] for all i
    vals_2x_plus = 2 * x + existing_arr
    for v in vals_2x_plus:
        if v in existing_sums_set:
            return True

    # Check x + 2*existing[i] for all i
    vals_x_plus_2 = x + 2 * existing_arr
    for v in vals_x_plus_2:
        if v in existing_sums_set:
            return True

    # Check x + existing[i] + existing[j] for i < j
    if n >= 2:
        ii, jj = np.triu_indices(n, k=1)
        pair_sums = x + existing_arr[ii] + existing_arr[jj]
        for v in pair_sums:
            if v in existing_sums_set:
                return True

    return False


def _update_sums_fast(existing_arr, existing_sums_set, x):
    """Add all new triple sums involving x to existing_sums_set."""
    n = len(existing_arr)
    existing_sums_set.add(3 * x)

    for i in range(n):
        existing_sums_set.add(2 * x + existing_arr[i])
        existing_sums_set.add(x + 2 * existing_arr[i])

    if n >= 2:
        ii, jj = np.triu_indices(n, k=1)
        pair_sums = x + existing_arr[ii] + existing_arr[jj]
        for v in pair_sums:
            existing_sums_set.add(int(v))


def _greedy_from_candidates(candidates, existing, existing_arr, existing_sums, existing_set):
    """Try to add candidates in order, returning updated structures and count added."""
    added = 0
    for x in candidates:
        if x in existing_set:
            continue
        if _check_collision_fast(existing_arr, existing_sums, x):
            continue
        _update_sums_fast(existing_arr, existing_sums, x)
        existing.append(x)
        existing_arr = np.append(existing_arr, x)
        existing_set.add(x)
        added += 1
    return existing, existing_arr, existing_sums, existing_set, added


def _construct_greedy(start_val, step_pattern, rng, deadline):
    """
    Greedy construction with a given starting value and step pattern.
    step_pattern is a callable that takes current x and returns next x.
    """
    selected = []
    selected_arr = np.array([], dtype=np.int64)
    selected_sums = set()
    selected_set = set()

    x = start_val
    while x <= 4096:
        if time.time() > deadline:
            break
        if _check_collision_fast(selected_arr, selected_sums, x):
            pass
        else:
            _update_sums_fast(selected_arr, selected_sums, x)
            selected.append(x)
            selected_arr = np.append(selected_arr, x)
            selected_set.add(x)
        x = step_pattern(x, rng)

    return selected


def construct():
    """
    Find a large B_3[1] set in [1, 4096] where all unordered triple sums
    (allowing repetitions) are distinct.
    """
    start_time = time.time()
    deadline = start_time + 70  # safety margin

    best = []
    rng = random.Random(42)

    # Strategy 1: Deterministic greedy from 1
    existing = []
    existing_arr = np.array([], dtype=np.int64)
    existing_sums = set()
    existing_set = set()
    for x in range(1, 4097):
        if time.time() > deadline:
            break
        if _check_collision_fast(existing_arr, existing_sums, x):
            continue
        _update_sums_fast(existing_arr, existing_sums, x)
        existing.append(x)
        existing_arr = np.append(existing_arr, x)
        existing_set.add(x)

    if len(existing) > len(best) and _is_valid_triples(existing):
        best = list(existing)

    # Strategy 2: Randomized greedy with various step patterns
    for attempt in range(200):
        if time.time() > deadline - 5:
            break

        start_val = rng.randint(1, 100)
        base_step = rng.randint(2, 8)
        jump_prob = rng.uniform(0.05, 0.25)

        def step_pattern(x, rng, base_step=base_step, jump_prob=jump_prob):
            if rng.random() < jump_prob:
                return x + base_step * rng.randint(2, 4)
            else:
                return x + base_step

        selected = _construct_greedy(start_val, step_pattern, rng, deadline - 5)
        if len(selected) > len(best):
            best = selected

    # Strategy 3: Local improvement - try to add more elements to best set
    if len(best) > 0 and time.time() < deadline - 10:
        current = list(best)
        current.sort()
        current_arr = np.array(current, dtype=np.int64)
        current_sums = set(_all_triple_sums(current))
        current_set = set(current)

        # Try to add elements in order
        for x in range(1, 4097):
            if time.time() > deadline - 10:
                break
            if x in current_set:
                continue
            if _check_collision_fast(current_arr, current_sums, x):
                continue
            _update_sums_fast(current_arr, current_sums, x)
            current.append(x)
            current_arr = np.append(current_arr, x)
            current_set.add(x)
            current.sort()
            current_arr = np.sort(current_arr)

        if len(current) > len(best):
            best = current

    # Strategy 4: Swap-based improvement
    if len(best) > 0 and time.time() < deadline - 5:
        current = list(best)
        current.sort()

        for iteration in range(30):
            if time.time() > deadline - 5:
                break

            if not _is_valid_triples(current):
                current = list(best)
                break

            current_arr = np.array(current, dtype=np.int64)
            current_sums = set(_all_triple_sums(current))
            current_set = set(current)
            n = len(current)

            # Try adding
            added = False
            for x in range(1, 4097):
                if x in current_set:
                    continue
                if _check_collision_fast(current_arr, current_sums, x):
                    continue
                _update_sums_fast(current_arr, current_sums, x)
                current.append(x)
                current_arr = np.append(current_arr, x)
                current_set.add(x)
                current.sort()
                current_arr = np.sort(current_arr)
                added = True
                break

            if not added and n > 1:
                # Remove a random element, try to find replacement
                idx = rng.randint(0, n - 1)
                removed_val = current.pop(idx)
                current_set.discard(removed_val)
                current.sort()
                current_arr = np.array(current, dtype=np.int64)
                current_sums = set(_all_triple_sums(current))

                for x in range(1, 4097):
                    if x in current_set:
                        continue
                    if _check_collision_fast(current_arr, current_sums, x):
                        continue
                    _update_sums_fast(current_arr, current_sums, x)
                    current.append(x)
                    current_arr = np.append(current_arr, x)
                    current_set.add(x)
                    current.sort()
                    current_arr = np.sort(current_arr)
                    break

                if len(current) > len(best):
                    best = list(current)

    # Final validation
    if best and not _is_valid_triples(best):
        # Fallback: try a simple valid set
        best = [1, 3, 9, 27, 81, 243, 729, 2187]

    return best


if __name__ == "__main__":
    result = construct()
    print(f"Found set of size {len(result)}")
    print(f"Set: {result[:20]}...")
    print(f"Valid: {_is_valid_triples(result)}")
