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

    def get_counts(mask):
        """Fast count of unordered pair sums (a<=b) in [128,256] using FFT."""
        m = len(mask)
        N = 1 << (2 * m - 1).bit_length()
        f = np.fft.rfft(mask.astype(np.float64), n=N)
        conv = np.fft.irfft(f * f, n=N)[:2 * m - 1]
        conv = np.rint(conv)
        counts = np.zeros(S, dtype=np.int64)
        for i, s in enumerate(range(min_sum, max_sum + 1)):
            ordered = int(conv[s])
            if s % 2 == 0:
                half = s // 2
                diag = 1 if (0 <= half <= n and mask[half]) else 0
            else:
                diag = 0
            counts[i] = (ordered + diag) // 2
        return counts

    def mask_to_list(mask):
        return [i for i in range(n + 1) if mask[i]]

    def list_to_mask(A_list):
        mask = np.zeros(n + 1, dtype=np.int64)
        for x in A_list:
            mask[x] = 1
        return mask

    def get_cost(mask):
        counts = get_counts(mask)
        if np.any(counts == 0):
            return float('inf')
        return int(np.max(counts))

    best_score = float('inf')
    best_mask = None

    def update_best(mask):
        nonlocal best_score, best_mask
        if not np.any(mask):
            return
        cost = get_cost(mask)
        if cost < best_score:
            best_score = cost
            best_mask = mask.copy()

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

    # 2. Iterative Reweighting (Multiplicative Weights Update)
    if best_mask is not None and time.time() < deadline - 35:
        try:
            # Start from best mask, convert to continuous weights
            x = best_mask.astype(np.float64).copy()
            # Add small noise to break symmetry
            x = np.clip(x + random.gauss(0, 0.05, len(x)), 0.01, 1.0)

            def reweight_iteration(x, lr=0.3, iterations_per_check=50):
                """Apply multiplicative weights update for several iterations."""
                for it in range(iterations_per_check):
                    if time.time() > deadline - 10:
                        break
                    # Threshold to binary for count computation
                    mask = (x > 0.5).astype(np.int64)
                    if not np.any(mask):
                        x = np.clip(x + 0.1, 0.01, 1.0)
                        continue

                    counts = get_counts(mask).astype(np.float64)

                    # Check for zeros - heavily penalize
                    zero_mask = (counts == 0)
                    if np.any(zero_mask):
                        # Find which elements contribute to zero sums
                        # For each zero sum s, we need pairs (a,b) with a+b=s
                        # Boost elements that could help cover these sums
                        for s_idx in np.where(zero_mask)[0]:
                            s = min_sum + s_idx
                            # For each element a in A, check if s-a is also in A or could be
                            # Simple heuristic: boost elements near s/2
                            target = s / 2.0
                            # Boost elements within a window around target
                            lo = max(0, int(target) - 10)
                            hi = min(n, int(target) + 10)
                            x[lo:hi+1] = np.clip(x[lo:hi+1] * (1 + lr), 0.01, 1.0)
                        continue

                    # All sums covered. Now balance the multiplicities.
                    # Compute gradient: for each element i, how much does it contribute to each sum?
                    # Element i contributes to sum s if there exists j in A such that i+j=s (i<=j for unordered)
                    # For simplicity, use a smoothed approach:
                    # weight[i] *= exp(-lr * sum_s max(0, counts[s] - target) * contrib[i,s])
                    # where contrib[i,s] = number of pairs (i,j) with i+j=s, j in A, i<=j

                    target_count = np.mean(counts)
                    excess = np.maximum(0, counts - target_count)  # excess load on each sum

                    # For each element i, compute its contribution to over-represented sums
                    # contrib[i] = sum over s where excess[s]>0 of (number of j in A with i+j=s, i<=j)
                    contrib = np.zeros(n + 1, dtype=np.float64)
                    for i in range(n + 1):
                        if x[i] < 0.01:
                            continue
                        for j in range(i, n + 1):  # i <= j for unordered
                            s = i + j
                            if min_sum <= s <= max_sum:
                                s_idx = s - min_sum
                                if excess[s_idx] > 0:
                                    contrib[i] += excess[s_idx]

                    # Multiplicative update: penalize elements contributing to over-represented sums
                    # Also, slightly boost elements that help under-represented sums
                    deficit = np.maximum(0, target_count - counts)
                    for i in range(n + 1):
                        if x[i] < 0.01:
                            continue
                        deficit_contrib = 0.0
                        for j in range(i, n + 1):
                            s = i + j
                            if min_sum <= s <= max_sum:
                                s_idx = s - min_sum
                                if deficit[s_idx] > 0:
                                    deficit_contrib += deficit[s_idx]

                        # Net penalty: excess contribution minus deficit contribution
                        net = contrib[i] - 0.5 * deficit_contrib
                        x[i] *= np.exp(-lr * net)
                        x[i] = np.clip(x[i], 0.01, 1.0)

                return x

            # Run reweighting iterations
            max_reweight_iters = 200
            for rw_iter in range(max_reweight_iters):
                if time.time() > deadline - 15:
                    break
                x = reweight_iteration(x, lr=0.2, iterations_per_check=10)
                # Periodically threshold and update best
                if rw_iter % 20 == 0 or rw_iter == max_reweight_iters - 1:
                    mask = (x > 0.5).astype(np.int64)
                    if np.any(mask):
                        update_best(mask)

            # Final threshold and update
            mask = (x > 0.5).astype(np.int64)
            if np.any(mask):
                update_best(mask)
        except Exception:
            pass

    # 3. Local Search Refinement
    if best_mask is None:
        best_mask = np.zeros(n + 1, dtype=np.int64)
        best_mask[64:129] = 1
        best_score = get_cost(best_mask)

    current_mask = best_mask.copy()
    current_cost = best_score

    def try_local_search(mask, cost, time_limit):
        nonlocal best_score, best_mask
        current_mask = mask.copy()
        current_cost = cost

        while time.time() < time_limit:
            improved = False

            # 1. Try removing (order by least impact)
            ones = [i for i in range(n + 1) if current_mask[i] == 1]
            random.shuffle(ones)
            for i in ones:
                if time.time() > time_limit:
                    break
                trial_mask = current_mask.copy()
                trial_mask[i] = 0
                if not np.any(trial_mask):
                    continue
                tc = get_cost(trial_mask)
                if tc <= current_cost:
                    current_mask = trial_mask
                    current_cost = tc
                    if tc < best_score:
                        best_score = tc
                        best_mask = current_mask.copy()
                    improved = True
                    break

            # 2. Try adding
            if not improved:
                zeros = [i for i in range(n + 1) if current_mask[i] == 0]
                random.shuffle(zeros)
                for i in zeros:
                    if time.time() > time_limit:
                        break
                    trial_mask = current_mask.copy()
                    trial_mask[i] = 1
                    tc = get_cost(trial_mask)
                    if tc < current_cost:
                        current_mask = trial_mask
                        current_cost = tc
                        if tc < best_score:
                            best_score = tc
                            best_mask = current_mask.copy()
                        improved = True
                        break

            # 3. Try swapping
            if not improved:
                ones = [i for i in range(n + 1) if current_mask[i] == 1]
                zeros = [i for i in range(n + 1) if current_mask[i] == 0]
                random.shuffle(ones)
                random.shuffle(zeros)
                if len(ones) > 25:
                    ones = ones[:25]
                if len(zeros) > 25:
                    zeros = zeros[:25]

                for i in ones:
                    if time.time() > time_limit:
                        break
                    for j in zeros:
                        if time.time() > time_limit:
                            break
                        trial_mask = current_mask.copy()
                        trial_mask[i] = 0
                        trial_mask[j] = 1
                        tc = get_cost(trial_mask)
                        if tc < current_cost:
                            current_mask = trial_mask
                            current_cost = tc
                            if tc < best_score:
                                best_score = tc
                                best_mask = current_mask.copy()
                            improved = True
                            break
                    if improved:
                        break

            # 4. Random perturbation to escape local minima
            if not improved and time.time() < time_limit - 3:
                flip_count = random.randint(2, 5)
                indices = random.sample(range(n + 1), flip_count)
                trial_mask = current_mask.copy()
                for idx in indices:
                    trial_mask[idx] = 1 - trial_mask[idx]
                if np.any(trial_mask):
                    tc = get_cost(trial_mask)
                    if tc < current_cost:
                        current_mask = trial_mask
                        current_cost = tc
                        if tc < best_score:
                            best_score = tc
                            best_mask = current_mask.copy()
                        improved = True
                    elif tc == current_cost and random.random() < 0.15:
                        current_mask = trial_mask
                        if tc < best_score:
                            best_score = tc
                            best_mask = current_mask.copy()
                        improved = True

            if not improved:
                break

    try_local_search(best_mask, best_score, deadline - 3)

    # Final check
    if best_mask is None or not np.any(best_mask):
        return list(range(64, 129))

    return mask_to_list(best_mask)
