import time
import random
import numpy as np


def construct():
    n = 256
    t0 = time.time()
    deadline = t0 + 55  # 5s safety margin before 60s kill

    # Precompute all arithmetic progressions
    # A progression is (start, step, length) with start in [0, n-1], step >= 1
    # Indices: start, start+step, ..., start+(length-1)*step, all < n
    progressions = []
    prog_by_index = [[] for _ in range(n)]

    pid = 0
    for step in range(1, n):
        for start in range(step):
            max_len = (n - 1 - start) // step + 1
            for length in range(1, max_len + 1):
                indices = tuple(start + k * step for k in range(length))
                progressions.append(indices)
                for idx in indices:
                    prog_by_index[idx].append(pid)
                pid += 1

    num_progs = len(progressions)

    # Precompute progression lengths for fast updates
    prog_lengths = np.array([len(p) for p in progressions], dtype=np.int16)

    best_signs = None
    best_max_abs = float('inf')

    # Multi-start simulated annealing
    num_starts = 5
    time_per_start = (deadline - t0) / num_starts

    for start_iter in range(num_starts):
        start_time = time.time()
        if start_time > deadline - 2:
            break

        rng = random.Random(42 + start_iter * 1000)
        signs = np.array([rng.choice((-1, 1)) for _ in range(n)], dtype=np.int8)

        # Compute initial sums
        sums = np.zeros(num_progs, dtype=np.int16)
        for pidx, indices in enumerate(progressions):
            s = 0
            for idx in indices:
                s += signs[idx]
            sums[pidx] = s

        max_abs = int(np.max(np.abs(sums)))

        # Simulated annealing parameters
        T = 5.0  # initial temperature
        T_min = 0.01
        cooling_rate = 0.995
        max_iters = 500000

        current_signs = signs.copy()
        current_sums = sums.copy()
        current_max = max_abs

        local_best_signs = signs.copy()
        local_best_max = max_abs

        iters = 0
        while iters < max_iters:
            if time.time() - start_time > time_per_start - 0.5:
                break
            if time.time() > deadline - 1:
                break

            iters += 1

            # Select a flip index: weighted by how "bad" the progressions are
            # Pick a random progression, weighted by |sum|
            abs_sums = np.abs(current_sums)
            # Use a simple approach: pick from top-k worst progressions
            k = min(10, num_progs)
            if k == num_progs:
                worst_indices = np.argsort(abs_sums)[-k:]
            else:
                worst_indices = np.argpartition(abs_sums, -k)[-k:]

            # Pick one of the worst progressions
            worst_pid = int(rng.choice(worst_indices))
            worst_indices_list = progressions[worst_pid]

            # Pick a random index from that progression
            flip_idx = rng.choice(worst_indices_list)

            # Try flipping
            old_sign = int(current_signs[flip_idx])
            new_sign = -old_sign
            delta = 2 * new_sign  # change in sum for each progression containing flip_idx

            # Update sums for progressions containing flip_idx
            affected = prog_by_index[flip_idx]
            for p in affected:
                current_sums[p] += delta

            new_max = int(np.max(np.abs(current_sums)))

            # Accept or reject
            if new_max <= local_best_max:
                # Always accept improvement or equal
                current_signs[flip_idx] = new_sign
                if new_max < local_best_max:
                    local_best_max = new_max
                    local_best_signs = current_signs.copy()
                current_max = new_max
            else:
                # Metropolis acceptance
                delta_cost = new_max - current_max
                if delta_cost < 0 or rng.random() < np.exp(-delta_cost / max(T, T_min)):
                    current_signs[flip_idx] = new_sign
                    current_max = new_max
                else:
                    # Revert
                    for p in affected:
                        current_sums[p] -= delta
                    current_signs[flip_idx] = old_sign

            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min

        # Update global best
        if local_best_max < best_max_abs:
            best_max_abs = local_best_max
            best_signs = local_best_signs.copy()

    if best_signs is None:
        # Fallback: random
        rng = random.Random(0)
        best_signs = np.array([rng.choice((-1, 1)) for _ in range(n)], dtype=np.int8)

    return [int(x) for x in best_signs]
