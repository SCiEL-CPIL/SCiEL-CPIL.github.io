"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 85.0
    best_set = []
    max_sum = 3 * 4096  # 12288

    # Strategy: greedy with NumPy boolean sum-lookup for O(1) conflict checks.
    # Multiple candidate orderings to escape local optima.

    for seed in range(500):
        if time.time() > deadline:
            break
        random.seed(seed)

        selected = []
        used = np.zeros(max_sum + 1, dtype=np.bool_)
        in_set = set()

        def try_add(x):
            # Triples involving x:
            # 1. (x,x,x) -> 3x
            # 2. (y,x,x) -> y + 2x   for y in selected
            # 3. (y,y,x) -> 2y + x   for y in selected
            # 4. (y,z,x) -> y + z + x for y < z in selected
            new_sums = [3 * x]
            for y in selected:
                new_sums.append(y + 2 * x)
                new_sums.append(2 * y + x)
            n = len(selected)
            for i in range(n):
                y = selected[i]
                for j in range(i + 1, n):
                    new_sums.append(y + selected[j] + x)

            # Check conflicts with existing sums
            arr = np.array(new_sums, dtype=np.int64)
            if used[arr].any():
                return False
            # Check internal duplicates among new sums
            if len(set(new_sums)) != len(new_sums):
                return False

            used[arr] = True
            selected.append(x)
            in_set.add(x)
            return True

        # Build candidate ordering based on seed
        order = seed % 4
        if order == 0:
            # Ascending
            candidates = list(range(1, 4097))
        elif order == 1:
            # Descending
            candidates = list(range(4096, 0, -1))
        elif order == 2:
            # Random shuffle
            candidates = list(range(1, 4097))
            random.shuffle(candidates)
        else:
            # Ascending with small random jitter in blocks
            candidates = list(range(1, 4097))
            # shuffle in blocks of 50
            for b in range(0, len(candidates), 50):
                random.shuffle(candidates[b:b+50])

        for x in candidates:
            if time.time() > deadline:
                break
            if x in in_set:
                continue
            if len(selected) >= 80:
                break
            try_add(x)

        if len(selected) > len(best_set):
            best_set = list(selected)
            if len(best_set) >= 80:
                break

    return best_set
# EVOLVE-BLOCK-END
