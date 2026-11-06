import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin to stay well under 90s kill

    N = 256

    # 1. Precompute ALL APs (length >= 1) and reverse mapping
    ap_indices = []
    aps_containing = [[] for _ in range(N)]

    # Iterate over all valid (start, step) pairs
    for d in range(1, N + 1):
        for s in range(1, N + 1):
            # Generate the base sequence of indices for this (s, d)
            base_indices = []
            curr = s - 1
            while curr < N:
                base_indices.append(curr)
                curr += d

            # Every prefix of base_indices is a valid AP
            for L in range(1, len(base_indices) + 1):
                idx_list = base_indices[:L]
                ap_idx = len(ap_indices)
                ap_indices.append(idx_list)
                for idx in idx_list:
                    aps_containing[idx].append(ap_idx)

    num_aps = len(ap_indices)

    # 2. Helper functions

    def thue_morse(N):
        """Generate Thue-Morse sequence as +/-1 signs."""
        seq = np.zeros(N, dtype=np.int8)
        for i in range(N):
            # Parity of number of 1-bits in binary representation of i
            # Use bit_count if available (Python 3.10+), else bin().count()
            try:
                parity = i.bit_count() & 1
            except AttributeError:
                parity = bin(i).count('1') & 1
            seq[i] = 1 if parity == 0 else -1
        return seq

    def compute_sums(signs):
        """Compute sums for all APs given a signs array. Uses vectorized approach where possible."""
        sums = np.zeros(num_aps, dtype=np.int16)
        for r, idx_list in enumerate(ap_indices):
            sums[r] = np.sum(signs[idx_list])
        return sums

    def greedy_optimize(s_init, time_budget, rng):
        """
        Greedy Coordinate Descent with occasional random restarts/perturbations.
        Returns best signs and best score.
        """
        s = s_init.copy()
        sums = compute_sums(s)
        current_max = int(np.max(np.abs(sums)))
        best_s = s.copy()
        best_score = current_max

        indices_list = list(range(N))

        # Track consecutive non-improving sweeps to detect local minima
        no_improve_sweeps = 0
        max_no_improve = 3

        while time.time() - start_time < time_budget:
            rng.shuffle(indices_list)
            improved_this_sweep = False

            for i in indices_list:
                if time.time() - start_time > time_budget:
                    break

                old_val = s[i]
                delta_val = -2 * old_val
                affected = aps_containing[i]

                # Save old sums for affected APs
                old_sums = sums[affected].copy()

                # Apply flip
                sums[affected] += delta_val
                s[i] = -old_val

                new_max = int(np.max(np.abs(sums)))

                if new_max < current_max:
                    current_max = new_max
                    improved_this_sweep = True
                    if current_max < best_score:
                        best_score = current_max
                        best_s = s.copy()
                elif new_max == current_max:
                    # Accept ties with some probability to explore
                    if rng.random() < 0.05:
                        improved_this_sweep = True
                    else:
                        # Revert
                        sums[affected] = old_sums
                        s[i] = old_val
                else:
                    # Revert
                    sums[affected] = old_sums
                    s[i] = old_val

            if not improved_this_sweep:
                no_improve_sweeps += 1
                if no_improve_sweeps >= max_no_improve:
                    break
            else:
                no_improve_sweeps = 0

        return best_s, best_score

    # 3. Main loop: multiple restarts with different initializations
    rng_global = random.Random(42)
    best_global_s = None
    best_global_score = float('inf')

    # Budget allocation:
    # - First run: Thue-Morse init (strong heuristic)
    # - Second run: Random init
    # - Remaining runs: Random inits, each with decreasing time budget
    # We want to spend most time on the best-performing init.

    # Run 1: Thue-Morse
    tm = thue_morse(N)
    t1 = time.time()
    s1, score1 = greedy_optimize(tm, time_limit * 0.4, rng_global)
    elapsed1 = time.time() - t1
    if score1 < best_global_score:
        best_global_score = score1
        best_global_s = s1.copy()

    # Run 2: Random init
    if time.time() - start_time < time_limit * 0.7:
        rand_init = np.array([rng_global.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
        t2 = time.time()
        s2, score2 = greedy_optimize(rand_init, time_limit * 0.3, rng_global)
        elapsed2 = time.time() - t2
        if score2 < best_global_score:
            best_global_score = score2
            best_global_s = s2.copy()

    # Remaining runs: more random inits with shorter budgets
    remaining_budget = time_limit - (time.time() - start_time)
    if remaining_budget > 2.0:
        # Do a few more short runs
        num_extra = min(5, int(remaining_budget / 3.0))
        for k in range(num_extra):
            if time.time() - start_time > time_limit - 1.0:
                break
            rand_init = np.array([rng_global.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
            t_k = time.time()
            s_k, score_k = greedy_optimize(rand_init, min(3.0, time_limit - (time.time() - start_time) - 1.0), rng_global)
            if score_k < best_global_score:
                best_global_score = score_k
                best_global_s = s_k.copy()

    # Final safety: if we haven't found anything (shouldn't happen), return random
    if best_global_s is None:
        best_global_s = np.array([rng_global.choice([-1, 1]) for _ in range(N)], dtype=np.int8)

    return [int(x) for x in best_global_s]
