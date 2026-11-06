import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    selected = []
    existing_sums = set()

    def try_add(x):
        nonlocal selected, existing_sums
        k = len(selected)
        if k == 0:
            existing_sums.add(3*x)
            selected.append(x)
            return True

        S = np.array(selected, dtype=np.int64)

        new_sums_list = [np.array([3*x], dtype=np.int64)]
        new_sums_list.append(np.array([2*x + s for s in S], dtype=np.int64))
        new_sums_list.append(np.array([x + 2*s for s in S], dtype=np.int64))

        # x + S[i] + S[j] for all i <= j
        pair_sums = S[:, None] + S[None, :]
        mask = np.triu(np.ones((k, k), dtype=bool))
        pair_upper = pair_sums[mask]
        new_sums_list.append(x + pair_upper)

        new_sums = np.concatenate(new_sums_list)

        # Check uniqueness among new sums
        if len(new_sums) != len(np.unique(new_sums)):
            return False

        # Check against existing
        new_sums_set = set(new_sums.tolist())
        if new_sums_set & existing_sums:
            return False

        existing_sums.update(new_sums_set)
        selected.append(x)
        return True

    # Phase 1: Greedy construction from small to large
    for x in range(1, 4097):
        if time.time() > deadline:
            break
        if len(selected) >= 80:
            break
        if try_add(x):
            continue
        if time.time() > deadline - 20:
            break

    # Phase 2: Try to fill remaining slots with random candidates
    if len(selected) < 80:
        remaining = 80 - len(selected)
        candidates = list(range(1, 4097))
        # Shuffle for randomness
        import random
        random.seed(42)
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            if len(selected) >= 80:
                break
            if try_add(x):
                continue

    return selected
