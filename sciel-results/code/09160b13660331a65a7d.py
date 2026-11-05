import time
import random
import numpy as np
from scipy.optimize import minimize, differential_evolution

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = 128
    min_sum = 128
    max_sum = 256
    S = max_sum - min_sum + 1  # 129 sums

    def get_counts_full(mask):
        """Fast count of unordered pair sums (a<=b) in [128,256] using FFT."""
        m = len(mask)
        N = 1 << (2 * m - 1).bit_length()
        f = np.fft.rfft(mask.astype(np.float64), n=N)
        conv = np.fft.irfft(f * f, n=N)[:2 * m - 1]
        conv = np.rint(conv).astype(np.int64)
        sums = np.arange(min_sum, max_sum + 1)
        ordered_counts = conv[sums]
        even_mask = (sums % 2 == 0)
        half_sums = sums // 2
        diag = np.where(even_mask, 
                        np.where((half_sums >= 0) & (half_sums <= n), 
                                mask[half_sums], 0), 
                        0)
        counts = (ordered_counts + diag) // 2
        return counts

    def get_cost_from_counts(counts):
        if np.any(counts == 0):
            return float('inf')
        return int(np.max(counts))

    def mask_to_list(mask):
        return [i for i in range(n + 1) if mask[i]]

    def list_to_mask(A_list):
        mask = np.zeros(n + 1, dtype=np.int64)
        for x in A_list:
            mask[x] = 1
        return mask

    def update_counts_add(mask, counts, i):
        """Update counts when adding element i to mask."""
        new_counts = counts.copy()
        for j in range(i, n + 1):
            if mask[j]:
                s = i + j
                if min_sum <= s <= max_sum:
                    new_counts[s - min_sum] += 1
        return new_counts

    def update_counts_remove(mask, counts, i):
        """Update counts when removing element i from mask."""
        new_counts = counts.copy()
        for j in range(i, n + 1):
            if mask[j]:
                s = i + j
                if min_sum <= s <= max_sum:
                    new_counts[s - min_sum] -= 1
        return new_counts

    best_score = float('inf')
    best_mask = None

    def update_best(mask, counts=None):
        nonlocal best_score, best_mask
        if not np.any(mask):
            return float('inf')
        if counts is None:
            counts = get_counts_full(mask)
        cost = get_cost_from_counts(counts)
        if cost < best_score:
            best_score = cost
            best_mask = mask.copy()
        return cost

    # 1. Heuristic Initializations
    for offset in range(0, 40):
        lo = max(0, 64 - offset)
        hi = min(128, 64 + offset)
        mask = np.zeros(n + 1, dtype=np.int64)
        mask[lo:hi + 1] = 1
        update_best(mask)

    for mod in [3, 4, 5, 6, 7, 8, 9, 10]:
        for keep in range(1, mod):
            mask = np.zeros(n + 1, dtype=np.int64)
            for x in range(n + 1):
                if x % mod != mod - keep:
                    mask[x] = 1
            update_best(mask)

    random.seed(42)
    for _ in range(80):
        if time.time() > deadline - 40:
            break
        size = random.randint(25, 65)
        A = random.sample(range(n + 1), size)
        mask = list_to_mask(A)
        update_best(mask)

    # 2. Local Search with incremental counts
    if best_mask is not None and time.time() < deadline - 5:
        current_mask = best_mask.copy()
        current_counts = get_counts_full(current_mask)
        current_cost = get_cost_from_counts(current_counts)

        no_improve = 0
        max_no_improve = 5
        time_limit = deadline - 3

        while time.time() < time_limit:
            improved = False

            # Phase 1: Try removals
            ones = [i for i in range(n + 1) if current_mask[i]]
            random.shuffle(ones)
            for i in ones:
                if time.time() > time_limit:
                    break
                trial_counts = update_counts_remove(current_mask, current_counts, i)
                if np.any(trial_counts < 0):
                    continue
                tc = get_cost_from_counts(trial_counts)
                if tc <= current_cost:
                    current_mask[i] = 0
                    current_counts = trial_counts
                    current_cost = tc
                    if tc < best_score:
                        best_score = tc
                        best_mask = current_mask.copy()
                    improved = True
                    no_improve = 0
                    break

            # Phase 2: Try additions
            if not improved:
                zeros = [i for i in range(n + 1) if not current_mask[i]]
                random.shuffle(zeros)
                for i in zeros:
                    if time.time() > time_limit:
                        break
                    trial_counts = update_counts_add(current_mask, current_counts, i)
                    tc = get_cost_from_counts(trial_counts)
                    if tc < current_cost:
                        current_mask[i] = 1
                        current_counts = trial_counts
                        current_cost = tc
                        if tc < best_score:
                            best_score = tc
                            best_mask = current_mask.copy()
                        improved = True
                        no_improve = 0
                        break

            # Phase 3: Try swaps
            if not improved:
                ones = [i for i in range(n + 1) if current_mask[i]]
                zeros = [i for i in range(n + 1) if not current_mask[i]]
                random.shuffle(ones)
                random.shuffle(zeros)
                for i in ones:
                    if time.time() > time_limit:
                        break
                    # Temporarily remove i
                    temp_mask = current_mask.copy()
                    temp_mask[i] = 0
                    temp_counts = update_counts_remove(current_mask, current_counts, i)
                    for j in zeros:
                        if time.time() > time_limit:
                            break
                        # Add j to the mask without i
                        trial_counts = update_counts_add(temp_mask, temp_counts, j)
                        if np.any(trial_counts < 0):
                            continue
                        tc = get_cost_from_counts(trial_counts)
                        if tc < current_cost:
                            current_mask[i] = 0
                            current_mask[j] = 1
                            current_counts = trial_counts
                            current_cost = tc
                            if tc < best_score:
                                best_score = tc
                                best_mask = current_mask.copy()
                            improved = True
                            no_improve = 0
                            break
                    if improved:
                        break

            # Phase 4: Perturbation
            if not improved:
                no_improve += 1
                if no_improve >= max_no_improve:
                    flip_count = random.randint(3, 7)
                else:
                    flip_count = random.randint(2, 4)
                indices = random.sample(range(n + 1), flip_count)
                trial_mask = current_mask.copy()
                for idx in indices:
                    trial_mask[idx] = 1 - trial_mask[idx]
                if np.any(trial_mask):
                    trial_counts = get_counts_full(trial_mask)
                    tc = get_cost_from_counts(trial_counts)
                    if tc < current_cost:
                        current_mask = trial_mask
                        current_counts = trial_counts
                        current_cost = tc
                        if tc < best_score:
                            best_score = tc
                            best_mask = current_mask.copy()
                        no_improve = 0
                    elif tc == current_cost and random.random() < 0.3:
                        current_mask = trial_mask
                        current_counts = trial_counts
                        no_improve = 0

            if not improved and no_improve >= max_no_improve + 3:
                break

    # Final polish: try to remove any redundant elements
    if best_mask is not None and time.time() < deadline - 1:
        final_mask = best_mask.copy()
        final_counts = get_counts_full(final_mask)
        ones = [i for i in range(n + 1) if final_mask[i]]
        random.shuffle(ones)
        for i in ones:
            if time.time() > deadline - 0.5:
                break
            trial_counts = update_counts_remove(final_mask, final_counts, i)
            if np.any(trial_counts < 0):
                continue
            tc = get_cost_from_counts(trial_counts)
            if tc <= best_score:
                final_mask[i] = 0
                final_counts = trial_counts
                if tc < best_score:
                    best_score = tc
                    best_mask = final_mask.copy()
        result_mask = final_mask
    else:
        result_mask = best_mask

    if result_mask is None or not np.any(result_mask):
        return list(range(64, 129))

    return mask_to_list(result_mask)
