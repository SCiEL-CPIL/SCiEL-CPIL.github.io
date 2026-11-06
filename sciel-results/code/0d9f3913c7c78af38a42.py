import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 150s kill limit

    N = 1000
    MAX_SUM = 2 * N  # 2000

    def greedy_build(seed=42, noise_scale=0.0):
        """Fast greedy using numpy sum-count array: O(1) validity per candidate."""
        rng = random.Random(seed)
        chosen = []
        chosen_set = set()
        # sum_counts[s] = number of pairs (a,b) with a<=b, a+b=s
        sum_counts = np.zeros(MAX_SUM + 1, dtype=np.int32)

        while True:
            if time.time() > deadline:
                break

            best_x = None
            best_tight = -1  # we want minimum tight count
            best_noise = 0.0

            for x in range(1, N + 1):
                if x in chosen_set:
                    continue

                # Check 2*x
                s2 = 2 * x
                if sum_counts[s2] >= 2:
                    continue

                # Check x + y for all y in chosen using vectorized numpy
                # sums = x + array_of_chosen
                if chosen:
                    chosen_arr = np.array(chosen, dtype=np.int32)
                    sums = x + chosen_arr
                    # Filter valid sums (<= MAX_SUM)
                    valid_mask = sums <= MAX_SUM
                    if not np.all(valid_mask):
                        sums = sums[valid_mask]
                    if len(sums) > 0:
                        sc = sum_counts[sums]
                        if np.any(sc >= 2):
                            continue

                # Count tight sums (count == 1)
                tight = 0
                s2 = 2 * x
                if sum_counts[s2] == 1:
                    tight += 1
                if chosen:
                    chosen_arr = np.array(chosen, dtype=np.int32)
                    sums = x + chosen_arr
                    valid_mask = sums <= MAX_SUM
                    if np.any(valid_mask):
                        sc = sum_counts[sums[valid_mask]]
                        tight += int(np.sum(sc == 1))

                noise = rng.random() * noise_scale
                score = tight + noise

                if best_x is None or score < best_tight + best_noise:
                    best_x = x
                    best_tight = tight
                    best_noise = noise

            if best_x is None:
                break

            # Add best_x
            chosen.append(best_x)
            chosen_set.add(best_x)

            # Update sum_counts
            s2 = 2 * best_x
            sum_counts[s2] += 1
            if chosen:
                # Get all other chosen elements
                others = [y for y in chosen if y != best_x]
                if others:
                    others_arr = np.array(others, dtype=np.int32)
                    sums = best_x + others_arr
                    valid_mask = sums <= MAX_SUM
                    if np.any(valid_mask):
                        sum_counts[sums[valid_mask]] += 1

        return chosen, sum_counts, chosen_set

    def can_add_fast(x, chosen_set, sum_counts):
        """O(1) check using numpy sum-count array."""
        if x in chosen_set:
            return False
        if sum_counts[2 * x] >= 2:
            return False
        # Need to check x + y for all y in chosen_set
        # This is still O(|chosen|) but with fast array lookup
        for y in chosen_set:
            s = x + y
            if s <= MAX_SUM and sum_counts[s] >= 2:
                return False
        return True

    def rebuild_sum_counts(chosen):
        """Rebuild sum_counts from scratch."""
        sum_counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
        for i, x in enumerate(chosen):
            sum_counts[2 * x] += 1
            for j in range(i + 1, len(chosen)):
                s = x + chosen[j]
                if s <= MAX_SUM:
                    sum_counts[s] += 1
        return sum_counts

    def local_search(chosen, max_iters=500):
        """Improve set by trying additions and swaps with numpy sum-counts."""
        chosen = list(chosen)
        chosen_set = set(chosen)
        sum_counts = rebuild_sum_counts(chosen)

        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1

            # Try to add any missing element (fast scan)
            added = False
            for x in range(1, N + 1):
                if x in chosen_set:
                    continue
                if sum_counts[2 * x] >= 2:
                    continue
                # Check all sums
                valid = True
                for y in chosen:
                    s = x + y
                    if s <= MAX_SUM and sum_counts[s] >= 2:
                        valid = False
                        break
                if valid:
                    chosen.append(x)
                    chosen_set.add(x)
                    sum_counts[2 * x] += 1
                    for y in chosen:
                        if y != x:
                            s = x + y
                            if s <= MAX_SUM:
                                sum_counts[s] += 1
                    improved = True
                    added = True
                    break

            if added:
                continue

            # No simple addition; try swaps
            # For each element to remove, check if we can add a different one
            found_swap = False
            for rem in chosen[:]:
                if time.time() > deadline:
                    break

                # Temporarily remove rem from sum_counts
                sum_counts[2 * rem] -= 1
                for z in chosen:
                    if z != rem:
                        s = rem + z
                        if s <= MAX_SUM:
                            sum_counts[s] -= 1

                temp_set = chosen_set - {rem}
                temp_chosen = [x for x in chosen if x != rem]

                # Try to find an element y not in temp_set that can be added
                found = False
                for y in range(1, N + 1):
                    if y in temp_set:
                        continue
                    if sum_counts[2 * y] >= 2:
                        continue
                    valid = True
                    for z in temp_chosen:
                        s = y + z
                        if s <= MAX_SUM and sum_counts[s] >= 2:
                            valid = False
                            break
                    if valid:
                        # Perform the swap
                        chosen_set.discard(rem)
                        chosen.remove(rem)
                        chosen.append(y)
                        chosen_set.add(y)
                        sum_counts[2 * y] += 1
                        for z in chosen:
                            if z != y:
                                s = y + z
                                if s <= MAX_SUM:
                                    sum_counts[s] += 1
                        improved = True
                        found = True
                        found_swap = True
                        break

                if found:
                    break

                # If no swap found, restore rem
                if not found:
                    sum_counts[2 * rem] += 1
                    for z in temp_chosen:
                        s = rem + z
                        if s <= MAX_SUM:
                            sum_counts[s] += 1

            if found_swap:
                continue

        return chosen, sum_counts, chosen_set

    best_set = []
    best_size = 0

    # Phase 1: Deterministic greedy with various seeds
    for seed in range(30):
        if time.time() > deadline:
            break
        chosen, _, _ = greedy_build(seed=seed, noise_scale=0.0)
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = chosen

    # Phase 2: Local search on best found
    if best_set and time.time() < deadline:
        best_set, _, _ = local_search(best_set, max_iters=300)
        best_size = len(best_set)

    # Phase 3: Randomized greedy restarts with local search
    while time.time() < deadline:
        seed = random.randint(0, 100000)
        noise = random.choice([0.0, 0.1, 0.3, 0.5, 1.0])
        chosen, _, _ = greedy_build(seed=seed, noise_scale=noise)

        # Run local search if it looks promising
        if len(chosen) >= best_size - 3 and time.time() < deadline:
            chosen, _, _ = local_search(chosen, max_iters=150)

        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = chosen

    return best_set
