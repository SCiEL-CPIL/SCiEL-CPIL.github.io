import random
import time


def max_valid_subset_size(arr):
    """Find the largest subset of arr where all unordered pair sums (i<=j) are distinct."""
    n = len(arr)
    best = 0

    def backtrack(start, current_subset, current_sums):
        nonlocal best
        if len(current_subset) > best:
            best = len(current_subset)
        for i in range(start, n):
            val = arr[i]
            if 2 * val in current_sums:
                continue
            valid = True
            new_sums = set()
            for existing in current_subset:
                s = existing + val
                if s in current_sums:
                    valid = False
                    break
                new_sums.add(s)
            if not valid:
                continue
            new_sums.add(2 * val)
            backtrack(i + 1, current_subset + [val], current_sums | new_sums)

    backtrack(0, [], set())
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 85

    # Structured initial candidates designed to maximize additive collisions
    candidates = [
        list(range(12)),
        [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14],
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16],
        [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13],
        [0, 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12],
        [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12],
    ]

    best = None
    best_score = 13

    for cand in candidates:
        if time.time() > deadline:
            break
        score = max_valid_subset_size(cand)
        if score < best_score:
            best_score = score
            best = list(cand)
        if best_score <= 3:
            return best

    if best is None:
        best = list(range(12))
        best_score = max_valid_subset_size(best)

    # Local search: mutate 1-3 elements at a time, keep if improves
    current = list(best)
    current_score = best_score
    rng = random.Random(42)

    for _ in range(50000):
        if time.time() > deadline:
            break
        num_changes = rng.randint(1, 3)
        indices = rng.sample(range(12), num_changes)
        old_vals = [current[i] for i in indices]
        for idx in indices:
            current[idx] = rng.randint(0, 128)

        if len(set(current)) != 12:
            for i, idx in enumerate(indices):
                current[idx] = old_vals[i]
            continue

        score = max_valid_subset_size(current)
        if score < current_score:
            current_score = score
            if score < best_score:
                best_score = score
                best = list(current)
                if best_score <= 3:
                    return best
        else:
            for i, idx in enumerate(indices):
                current[idx] = old_vals[i]

    return best
