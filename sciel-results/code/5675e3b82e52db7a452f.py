"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _max_sidon_size(arr):
    """Return the size of the largest Sidon subset of arr (pair sums i<=j unique)."""
    n = len(arr)
    arr = sorted(arr)
    best = 0
    # Iterate subsets in descending order of size for early termination
    from itertools import combinations
    for k in range(n, best, -1):
        found = False
        for combo in combinations(range(n), k):
            subset = [arr[i] for i in combo]
            sums = set()
            is_sidon = True
            for i in range(k):
                for j in range(i, k):
                    s = subset[i] + subset[j]
                    if s in sums:
                        is_sidon = False
                        break
                    sums.add(s)
                if not is_sidon:
                    break
            if is_sidon:
                found = True
                break
        if found:
            best = k
            break
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    random.seed(42)

    # Start with a structured initial guess
    # Use a set with many additive collisions
    current = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
    current_val = _max_sidon_size(current)

    best = current[:]
    best_val = current_val

    # Local search: perturb one element at a time
    for _ in range(50000):
        if time.time() >= deadline:
            break
        idx = random.randint(0, 11)
        new_val = random.randint(0, 128)
        if new_val in current and new_val != current[idx]:
            continue
        old = current[idx]
        current[idx] = new_val
        val = _max_sidon_size(current)
        if val < best_val:
            best_val = val
            best = current[:]
        elif val > current_val or random.random() < 0.05:
            # Keep if better or random acceptance
            pass
        else:
            current[idx] = old
            continue
        current_val = val

    return best
# EVOLVE-BLOCK-END
