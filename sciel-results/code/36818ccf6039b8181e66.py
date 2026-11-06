import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 150s kill limit

    N = 1000
    MAX_SUM = 2 * N  # max possible sum is 2000

    def build_counts(chosen):
        """Build sum-count dict from a list of chosen elements."""
        counts = defaultdict(int)
        for i, x in enumerate(chosen):
            counts[2 * x] += 1
            for j in range(i + 1, len(chosen)):
                counts[x + chosen[j]] += 1
        return counts

    def greedy_build(seed=42, noise_scale=0.0):
        """Fast greedy using numpy for scoring."""
        random.seed(seed)
        np.random.seed(seed)
        chosen = []
        chosen_set = set()
        counts_arr = np.zeros(MAX_SUM + 1, dtype=np.int32)
        available = np.ones(N + 1, dtype=bool)
        available[0] = False

        while True:
            if time.time() > deadline:
                break

            if len(chosen) == 0:
                if noise_scale > 0:
                    best_x = random.randint(1, N)
                else:
                    best_x = 1
                chosen.append(best_x)
                chosen_set.add(best_x)
                available[best_x] = False
                counts_arr[2 * best_x] += 1
                continue

            avail_indices = np.where(available[1:])[0] + 1
            if len(avail_indices) == 0:
                break

            doubles = 2 * avail_indices
            double_counts = counts_arr[doubles]
            invalid_double = double_counts >= 2
            tight_double = (double_counts == 1).astype(np.int32)

            invalid_mask = invalid_double.copy()
            tight_mask = tight_double.copy()

            for y in chosen:
                sums = avail_indices + y
                valid_sums = (sums >= 1) & (sums <= MAX_SUM)
                if not np.all(valid_sums):
                    sums = sums[valid_sums]
                    invalid_mask = invalid_mask[valid_sums]
                    tight_mask = tight_mask[valid_sums]
                
                if len(sums) == 0:
                    break
                    
                sum_counts = counts_arr[sums]
                new_invalid = sum_counts >= 2
                new_tight = (sum_counts == 1).astype(np.int32)
                
                invalid_mask = invalid_mask | new_invalid
                tight_mask = tight_mask + new_tight

            valid_mask = ~invalid_mask
            if not np.any(valid_mask):
                break

            valid_indices = avail_indices[valid_mask]
            valid_tight = tight_mask[valid_mask]

            if noise_scale > 0:
                noise = np.random.random(len(valid_indices)) * noise_scale
                scores = valid_tight + noise
            else:
                scores = valid_tight.astype(np.float64)

            best_idx = np.argmin(scores)
            best_x = int(valid_indices[best_idx])

            chosen.append(best_x)
            chosen_set.add(best_x)
            available[best_x] = False

            counts_arr[2 * best_x] += 1
            for y in chosen:
                if y != best_x:
                    counts_arr[best_x + y] += 1

        return chosen, build_counts(chosen), chosen_set

    def try_add_element(x, chosen_set, counts_arr):
        """Try to add element x if valid. Returns True if added."""
        if x in chosen_set:
            return False
        if counts_arr[2 * x] >= 2:
            return False
        for y in chosen_set:
            if counts_arr[x + y] >= 2:
                return False
        # Add it
        chosen_set.add(x)
        counts_arr[2 * x] += 1
        for y in list(chosen_set):
            if y != x:
                counts_arr[x + y] += 1
        return True

    def local_search(chosen, counts, chosen_set, max_iters=200):
        """Improve set with additions, swaps, and kicks."""
        chosen = list(chosen)
        chosen_set = set(chosen)
        counts_arr = np.zeros(MAX_SUM + 1, dtype=np.int32)
        for s, c in counts.items():
            counts_arr[s] = c

        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1

            # Phase A: Try to add any missing element
            for x in range(1, N + 1):
                if x in chosen_set:
                    continue
                if counts_arr[2 * x] >= 2:
                    continue
                valid = True
                for y in chosen_set:
                    if counts_arr[x + y] >= 2:
                        valid = False
                        break
                if valid:
                    chosen.append(x)
                    chosen_set.add(x)
                    counts_arr[2 * x] += 1
                    for y in chosen:
                        if y != x:
                            counts_arr[x + y] += 1
                    improved = True
                    break

            if improved:
                continue

            # Phase B: Try swaps - remove one, add best replacement
            best_swap = None
            best_swap_score = float('inf')
            
            for rem in chosen[:]:
                if time.time() > deadline:
                    break

                temp_counts_arr = counts_arr.copy()
                temp_counts_arr[2 * rem] -= 1
                for z in chosen:
                    if z != rem:
                        temp_counts_arr[rem + z] -= 1

                temp_set = chosen_set - {rem}

                for y in range(1, N + 1):
                    if y in temp_set:
                        continue
                    if temp_counts_arr[2 * y] >= 2:
                        continue
                    tight = 0
                    valid = True
                    for z in temp_set:
                        if temp_counts_arr[y + z] >= 2:
                            valid = False
                            break
                        if temp_counts_arr[y + z] == 1:
                            tight += 1
                    if not valid:
                        continue
                    if tight < best_swap_score:
                        best_swap_score = tight
                        best_swap = (rem, y, temp_counts_arr)

            if best_swap is not None:
                rem, y, temp_counts_arr = best_swap
                # Apply swap
                counts_arr[2 * rem] -= 1
                for z in chosen:
                    if z != rem:
                        counts_arr[rem + z] -= 1
                chosen.remove(rem)
                chosen_set.remove(rem)
                chosen.append(y)
                chosen_set.add(y)
                counts_arr[2 * y] += 1
                for z in chosen:
                    if z != y:
                        counts_arr[y + z] += 1
                improved = True
                continue

            # Phase C: Kick - remove 1-3 random elements, then greedily refill
            if iters % 3 == 0:  # Do kicks periodically
                kick_size = random.randint(1, 3)
                kick_candidates = random.sample(chosen, min(kick_size, len(chosen)))
                
                # Remove kick elements
                temp_chosen = [x for x in chosen if x not in kick_candidates]
                temp_set = set(temp_chosen)
                temp_counts_arr = np.zeros(MAX_SUM + 1, dtype=np.int32)
                for i, x in enumerate(temp_chosen):
                    temp_counts_arr[2 * x] += 1
                    for j in range(i + 1, len(temp_chosen)):
                        temp_counts_arr[x + temp_chosen[j]] += 1

                # Greedily refill
                for x in range(1, N + 1):
                    if x in temp_set:
                        continue
                    if temp_counts_arr[2 * x] >= 2:
                        continue
                    valid = True
                    for z in temp_set:
                        if temp_counts_arr[x + z] >= 2:
                            valid = False
                            break
                    if valid:
                        temp_chosen.append(x)
                        temp_set.add(x)
                        temp_counts_arr[2 * x] += 1
                        for z in temp_chosen:
                            if z != x:
                                temp_counts_arr[x + z] += 1

                if len(temp_chosen) > len(chosen):
                    chosen = temp_chosen
                    chosen_set = temp_set
                    counts_arr = temp_counts_arr
                    improved = True

        return chosen, build_counts(chosen), chosen_set

    def polish(chosen, counts, chosen_set, max_rounds=50):
        """Final polish: repeatedly try to add any element."""
        chosen = list(chosen)
        chosen_set = set(chosen)
        counts_arr = np.zeros(MAX_SUM + 1, dtype=np.int32)
        for s, c in counts.items():
            counts_arr[s] = c

        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            added = False
            for x in range(1, N + 1):
                if x in chosen_set:
                    continue
                if counts_arr[2 * x] >= 2:
                    continue
                valid = True
                for y in chosen_set:
                    if counts_arr[x + y] >= 2:
                        valid = False
                        break
                if valid:
                    chosen.append(x)
                    chosen_set.add(x)
                    counts_arr[2 * x] += 1
                    for y in chosen:
                        if y != x:
                            counts_arr[x + y] += 1
                    added = True
                    break
            if not added:
                break

        return chosen, build_counts(chosen), chosen_set

    best_set = []
    best_size = 0

    # Phase 1: Deterministic greedy with various seeds
    for seed in range(20):
        if time.time() > deadline:
            break
        chosen, counts, chosen_set = greedy_build(seed=seed, noise_scale=0.0)
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = chosen

    # Phase 2: Local search on best found
    if best_set and time.time() < deadline:
        counts = build_counts(best_set)
        chosen_set = set(best_set)
        best_set, counts, chosen_set = local_search(best_set, counts, chosen_set, max_iters=150)
        best_size = len(best_set)

    # Phase 3: Randomized greedy restarts with local search
    while time.time() < deadline - 20:  # Leave time for final polish
        seed = random.randint(0, 100000)
        noise = random.choice([0.0, 0.1, 0.3, 0.5, 1.0])
        chosen, counts, chosen_set = greedy_build(seed=seed, noise_scale=noise)

        if len(chosen) >= best_size - 5 and time.time() < deadline - 20:
            counts = build_counts(chosen)
            chosen_set = set(chosen)
            chosen, counts, chosen_set = local_search(chosen, counts, chosen_set, max_iters=100)

        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = chosen

    # Phase 4: Final polish
    if best_set and time.time() < deadline:
        counts = build_counts(best_set)
        chosen_set = set(best_set)
        best_set, counts, chosen_set = polish(best_set, counts, chosen_set, max_rounds=100)
        best_size = len(best_set)

    return best_set
