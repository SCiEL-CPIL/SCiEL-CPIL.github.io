import time
import random
import numpy as np
from scipy.optimize import differential_evolution

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s kill

    N = 128
    min_sum = 128
    max_sum = 256
    target_sums = list(range(min_sum, max_sum + 1))
    num_targets = len(target_sums)  # 129

    def get_counts_fast(A):
        """Returns array of counts for sums in [0, 256] using numpy"""
        A_arr = np.array(sorted(A), dtype=np.int64)
        k = len(A_arr)
        if k == 0:
            return np.zeros(max_sum + 1, dtype=np.int64)
        A_i = A_arr[:, None]
        A_j = A_arr[None, :]
        sums = A_i + A_j
        mask = np.triu(np.ones((k, k), dtype=bool), k=0)
        valid_sums = sums[mask]
        valid_sums = valid_sums[valid_sums <= max_sum]
        if len(valid_sums) > 0:
            counts = np.bincount(valid_sums, minlength=max_sum + 1).astype(np.int64)
        else:
            counts = np.zeros(max_sum + 1, dtype=np.int64)
        return counts

    def get_cost(A):
        counts = get_counts_fast(A)
        target_counts = counts[min_sum:max_sum + 1]
        if np.any(target_counts == 0):
            uncovered = np.sum(target_counts == 0)
            return 1000 + uncovered
        return int(np.max(target_counts))

    def get_cost_detailed(A):
        counts = get_counts_fast(A)
        target_counts = counts[min_sum:max_sum + 1]
        uncovered = np.sum(target_counts == 0)
        if uncovered > 0:
            return (1000 + uncovered, uncovered, counts)
        return (int(np.max(target_counts)), 0, counts)

    # DE fitness function: operates on binary vector x of length 129
    def de_fitness(x):
        A_set = set(int(i) for i in range(129) if x[i] > 0.5)
        if len(A_set) == 0:
            return 1000.0
        counts = get_counts_fast(A_set)
        target_counts = counts[min_sum:max_sum + 1]
        uncovered = np.sum(target_counts == 0)
        if uncovered > 0:
            return 1000.0 + float(uncovered)
        return float(np.max(target_counts))

    # Fast fitness for local search (vectorized incremental)
    def fast_fitness_from_vector(x):
        A_set = set(int(i) for i in range(129) if x[i] > 0.5)
        if len(A_set) == 0:
            return 1000.0
        counts = get_counts_fast(A_set)
        target_counts = counts[min_sum:max_sum + 1]
        uncovered = np.sum(target_counts == 0)
        if uncovered > 0:
            return 1000.0 + float(uncovered)
        return float(np.max(target_counts))

    # Phase 1: Differential Evolution
    best_A = None
    best_cost = float('inf')

    # Try a few DE runs with different seeds and bounds
    # Use a reasonable number of elements: between 15 and 30
    for de_seed in [42, 123, 777]:
        if time.time() > deadline - 20.0:
            break

        rng_de = np.random.RandomState(de_seed)

        # Initialize: start from a structured set
        # Try a few initial populations
        for init_mode in range(3):
            if time.time() > deadline - 15.0:
                break

            # Create initial binary vector
            if init_mode == 0:
                # Evenly spread
                k = 18
                base_points = np.linspace(0, 128, k, dtype=int)
                x_init = np.zeros(129)
                for p in base_points:
                    x_init[int(p)] = 1.0
            elif init_mode == 1:
                # Two clusters
                x_init = np.zeros(129)
                for i in range(10):
                    x_init[i] = 1.0
                for i in range(119, 129):
                    x_init[i] = 1.0
            else:
                # Random
                k = 20
                chosen = rng_de.choice(129, size=k, replace=False)
                x_init = np.zeros(129)
                for c in chosen:
                    x_init[c] = 1.0

            # Run DE with bounded number of iterations
            # DE will modify x; we need to handle the binary nature
            # Use a wrapper that rounds to binary after each evaluation
            def de_objective(x_continuous):
                # Round to binary
                x_bin = (x_continuous > 0.5).astype(float)
                return de_fitness(x_bin)

            # DE bounds: each variable in [0, 1]
            bounds = [(0.0, 1.0)] * 129

            try:
                result = differential_evolution(
                    de_objective,
                    bounds,
                    seed=de_seed + init_mode,
                    maxiter=50,
                    popsize=15,
                    tol=0.01,
                    mutation=(0.3, 0.9),
                    recombination=0.8,
                    polish=True,
                    init='sobol',
                    update_interval=1,
                    callback=None,
                    workers=1
                )
            except Exception:
                continue

            # Extract best from DE result
            x_best = (result.x > 0.5).astype(float)
            A_candidate = set(int(i) for i in range(129) if x_best[i] > 0.5)
            if len(A_candidate) > 0:
                cost, _, _ = get_cost_detailed(A_candidate)
                if cost < best_cost:
                    best_cost = cost
                    best_A = set(A_candidate)

    # Phase 2: Local search polish
    if best_A is not None:
        A = set(best_A)
        current_counts = get_counts_fast(A)
        current_cost, _, _ = get_cost_detailed(A)
        best_A = set(A)
        best_cost = current_cost

        rng = random.Random(12345)
        outside = list(set(range(129)) - A)
        A_list = list(A)

        def incremental_swap_cost(A_set, x_in, x_out, current_counts):
            new_counts = current_counts.copy()
            s = 2 * x_in
            if s <= max_sum:
                new_counts[s] -= 1
            for y in A_set:
                if y == x_in:
                    continue
                s = x_in + y
                if s <= max_sum:
                    new_counts[s] -= 1
            s = 2 * x_out
            if s <= max_sum:
                new_counts[s] += 1
            for y in A_set:
                if y == x_in:
                    continue
                s = x_out + y
                if s <= max_sum:
                    new_counts[s] += 1
            target_counts = new_counts[min_sum:max_sum + 1]
            uncovered = np.sum(target_counts == 0)
            if uncovered > 0:
                new_cost = 1000 + uncovered
            else:
                new_cost = int(np.max(target_counts))
            return new_counts, new_cost, uncovered

        def incremental_add_cost(A_set, x_add, current_counts):
            new_counts = current_counts.copy()
            s = 2 * x_add
            if s <= max_sum:
                new_counts[s] += 1
            for y in A_set:
                s = x_add + y
                if s <= max_sum:
                    new_counts[s] += 1
            target_counts = new_counts[min_sum:max_sum + 1]
            uncovered = np.sum(target_counts == 0)
            if uncovered > 0:
                new_cost = 1000 + uncovered
            else:
                new_cost = int(np.max(target_counts))
            return new_counts, new_cost, uncovered

        def incremental_remove_cost(A_set, x_rem, current_counts):
            new_counts = current_counts.copy()
            s = 2 * x_rem
            if s <= max_sum:
                new_counts[s] -= 1
            for y in A_set:
                if y == x_rem:
                    continue
                s = x_rem + y
                if s <= max_sum:
                    new_counts[s] -= 1
            target_counts = new_counts[min_sum:max_sum + 1]
            uncovered = np.sum(target_counts == 0)
            if uncovered > 0:
                new_cost = 1000 + uncovered
            else:
                new_cost = int(np.max(target_counts))
            return new_counts, new_cost, uncovered

        # Simulated annealing-style local search
        temperature = 5.0
        cooling = 0.9995
        min_temp = 0.1

        for iteration in range(200000):
            if time.time() > deadline - 3.0:
                break
            if temperature < min_temp:
                break

            move_type = rng.random()

            if move_type < 0.65 and len(outside) > 0:
                x_in = rng.choice(A_list)
                x_out = rng.choice(outside)
                new_counts, new_cost, _ = incremental_swap_cost(A, x_in, x_out, current_counts)
                delta = new_cost - current_cost
                if delta <= 0 or rng.random() < np.exp(-delta / temperature):
                    A.remove(x_in)
                    A.add(x_out)
                    outside.remove(x_out)
                    outside.append(x_in)
                    A_list.remove(x_in)
                    A_list.append(x_out)
                    current_counts = new_counts
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(A)
                        if best_cost <= 2:
                            break

            elif move_type < 0.85 and len(A) > 14:
                x = rng.choice(A_list)
                new_counts, new_cost, _ = incremental_remove_cost(A, x, current_counts)
                delta = new_cost - current_cost
                if delta <= 0 or rng.random() < np.exp(-delta / temperature):
                    A.remove(x)
                    outside.append(x)
                    A_list.remove(x)
                    current_counts = new_counts
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(A)
                        if best_cost <= 2:
                            break

            elif len(A) < 35:
                x = rng.choice(outside)
                new_counts, new_cost, _ = incremental_add_cost(A, x, current_counts)
                delta = new_cost - current_cost
                if delta <= 0 or rng.random() < np.exp(-delta / temperature):
                    A.add(x)
                    outside.remove(x)
                    A_list.append(x)
                    current_counts = new_counts
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(A)
                        if best_cost <= 2:
                            break

            temperature *= cooling

    # Final verification
    if best_A is None or len(best_A) == 0:
        best_A = set(range(0, 129, 8))

    return list(best_A)
