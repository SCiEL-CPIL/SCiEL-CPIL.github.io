import numpy as np
import time
import random
import sys

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin

    # 1. Generate all APs of length >= 2
    ap_indices = []
    for start in range(n):
        for step in range(1, n):
            terms = list(range(start, n, step))
            if len(terms) >= 2:
                ap_indices.append(terms)
    num_aps = len(ap_indices)

    # Precompute inverse mapping: pos -> list of ap_ids
    pos_to_aps = [[] for _ in range(n)]
    for ap_id, idx in enumerate(ap_indices):
        for i in idx:
            pos_to_aps[i].append(ap_id)

    # 2. Helper: Compute AP sums from a sign vector
    def compute_sums(signs):
        sums = np.zeros(num_aps, dtype=np.int32)
        for ap_id, idx in enumerate(ap_indices):
            sums[ap_id] = np.sum(signs[idx])
        return sums

    # 3. Histogram-based local search (fast)
    def local_search(signs, max_iters, deadline, rng, temp=0.0):
        signs = signs.copy()
        ap_sums = compute_sums(signs)
        current_max = int(np.max(np.abs(ap_sums)))
        best_signs = signs.copy()
        best_max = current_max

        # Build histogram of |ap_sums| for fast max tracking
        # ap_sums can range from -max_len to +max_len, max_len <= n
        max_abs = n + 1
        hist = np.zeros(2 * max_abs + 1, dtype=np.int64)
        for s in ap_sums:
            hist[s + max_abs] += 1

        def get_max_from_hist():
            # Find largest |s| with hist[s+max_abs] > 0
            for d in range(max_abs, -1, -1):
                if hist[d + max_abs] > 0 or hist[-d + max_abs] > 0:
                    return d
            return 0

        no_improve = 0
        for _ in range(max_iters):
            if time.time() > deadline:
                break

            i = rng.randint(0, n - 1)
            old_sign = signs[i]
            new_sign = -old_sign
            delta = 2 * new_sign  # +2 or -2

            affected = pos_to_aps[i]

            # Update sums and histogram
            for ap_id in affected:
                old_s = ap_sums[ap_id]
                new_s = old_s + delta
                hist[old_s + max_abs] -= 1
                ap_sums[ap_id] = new_s
                hist[new_s + max_abs] += 1

            new_max = get_max_from_hist()

            accept = False
            if new_max < current_max:
                accept = True
            elif new_max == current_max:
                if temp > 0 and rng.random() < 0.15:
                    accept = True
            else:
                if temp > 0.05:
                    delta_max = new_max - current_max
                    if rng.random() < np.exp(-delta_max / temp):
                        accept = True

            if accept:
                signs[i] = new_sign
                current_max = new_max
                no_improve = 0
                if current_max < best_max:
                    best_max = current_max
                    best_signs = signs.copy()
            else:
                # Revert
                for ap_id in affected:
                    new_s = ap_sums[ap_id]
                    old_s = new_s - delta
                    hist[new_s + max_abs] -= 1
                    ap_sums[ap_id] = old_s
                    hist[old_s + max_abs] += 1
                no_improve += 1

            if no_improve > 100:
                break

        return best_signs, best_max

    # 4. Spectral/subgradient approach: minimize sum of squares of AP sums
    #    x is a continuous vector in [-1, 1], project to ±1
    def spectral_approx(deadline):
        # Build incidence: for each position, which APs it belongs to
        # We'll do subgradient descent on f(x) = sum_j (sum_i in AP_j x_i)^2
        # Gradient: grad_i = 2 * sum_{j: i in AP_j} sum_j
        # Use a few random restarts with different initializations

        best_proj_signs = None
        best_proj_max = n + 1

        num_restarts = 5
        for r in range(num_restarts):
            if time.time() > deadline - 15.0:
                break

            rng = random.Random(1000 + r * 7919)

            # Initialize: random continuous vector
            x = np.array([rng.uniform(-1, 1) for _ in range(n)], dtype=np.float64)

            # Subgradient descent with decreasing step size
            num_steps = 800
            for step in range(num_steps):
                if time.time() > deadline - 15.0:
                    break

                # Compute AP sums
                ap_sums = np.zeros(num_aps, dtype=np.float64)
                for ap_id, idx in enumerate(ap_indices):
                    ap_sums[ap_id] = np.sum(x[idx])

                # Gradient: for each position, sum of 2*ap_sum over APs containing it
                grad = np.zeros(n, dtype=np.float64)
                for ap_id in range(num_aps):
                    s = 2.0 * ap_sums[ap_id]
                    for i in ap_indices[ap_id]:
                        grad[i] += s

                # Step size: decrease over time
                alpha = 2.0 / (1.0 + 0.01 * step)

                # Update
                x -= alpha * grad

                # Project to [-1, 1]
                np.clip(x, -1.0, 1.0, out=x)

                # Periodically project to ±1 and check
                if step % 50 == 49 or step == num_steps - 1:
                    proj = np.where(x >= 0, 1, -1).astype(np.int8)
                    ap_sums_int = compute_sums(proj)
                    proj_max = int(np.max(np.abs(ap_sums_int)))
                    if proj_max < best_proj_max:
                        best_proj_max = proj_max
                        best_proj_signs = proj.copy()

        return best_proj_signs, best_proj_max

    # 5. Main loop
    best_overall_signs = None
    best_overall_max = n + 1

    # Phase 1: Spectral approximation to find good basin
    if time.time() < deadline - 20.0:
        spec_signs, spec_max = spectral_approx(deadline)
        if spec_signs is not None:
            best_overall_signs = spec_signs
            best_overall_max = spec_max

    # Phase 2: Also try Thue-Morse as initialization
    thue_morse = np.array([1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)], dtype=np.int8)
    tm_signs, tm_max = local_search(thue_morse, 2000, deadline, random.Random(42), temp=0.0)
    if tm_max < best_overall_max:
        best_overall_max = tm_max
        best_overall_signs = tm_signs.copy()

    # Phase 3: Multiple random restarts with local search
    seeds = [42, 123, 456, 789, 1000, 2024, 3141, 2718, 9999, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 10101, 20202, 31337]

    for seed in seeds:
        if time.time() > deadline - 10.0:
            break

        rng = random.Random(seed)
        signs = np.array([rng.choice([-1, 1]) for _ in range(n)], dtype=np.int8)

        # Use small temperature for occasional random walks
        temp = 0.1 if seed % 3 == 0 else 0.0
        best_signs, best_max = local_search(signs, 3000, deadline, rng, temp=temp)

        if best_max < best_overall_max:
            best_overall_max = best_max
            best_overall_signs = best_signs.copy()

        # If we hit 2 or below, try harder
        if best_overall_max <= 2:
            rng2 = random.Random(seed * 7919 + 1)
            best_signs2, best_max2 = local_search(best_overall_signs, 5000, deadline, rng2, temp=0.05)
            if best_max2 < best_overall_max:
                best_overall_max = best_max2
                best_overall_signs = best_signs2.copy()

    # Phase 4: Final polish on best solution with strict greedy
    if best_overall_signs is not None and time.time() < deadline - 2.0:
        rng3 = random.Random(31337)
        final_signs, final_max = local_search(best_overall_signs, 15000, deadline, rng3, temp=0.0)
        if final_max < best_overall_max:
            best_overall_max = final_max
            best_overall_signs = final_signs

    # Phase 5: One more round of spectral + polish if time remains
    if time.time() < deadline - 25.0 and best_overall_max > 2:
        spec_signs2, spec_max2 = spectral_approx(deadline)
        if spec_signs2 is not None and spec_max2 < best_overall_max:
            best_overall_max = spec_max2
            best_overall_signs = spec_signs2.copy()
            # Polish it
            if time.time() < deadline - 3.0:
                rng4 = random.Random(99999)
                pol_signs, pol_max = local_search(best_overall_signs, 10000, deadline, rng4, temp=0.0)
                if pol_max < best_overall_max:
                    best_overall_max = pol_max
                    best_overall_signs = pol_signs

    if best_overall_signs is None:
        rng = random.Random(42)
        return [rng.choice([-1, 1]) for _ in range(n)]

    return [int(s) for s in best_overall_signs]
