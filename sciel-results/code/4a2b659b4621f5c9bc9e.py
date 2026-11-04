import numpy as np
import time
import itertools

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return list(itertools.product(range(q), repeat=n))

    total_points = q ** n  # 6561
    ball_size = 1 + n * (q - 1)  # 17

    # Pre-compute powers for fast index arithmetic
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Pre-compute neighbors (the ball of radius 1) for each index
    neighbors = np.empty((total_points, ball_size), dtype=np.int32)
    for idx in range(total_points):
        temp_idx = idx
        word = [0] * n
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q

        neighbors[idx, 0] = idx
        neighbor_idx = 1
        for pos in range(n):
            orig_digit = word[pos]
            power = powers[pos]

            # +1
            new_digit = (orig_digit + 1) % q
            if new_digit == 0:
                delta = -2 * power
            else:
                delta = power
            n_idx = idx + delta
            neighbors[idx, neighbor_idx] = n_idx
            neighbor_idx += 1

            # -1
            new_digit = (orig_digit - 1) % q
            if new_digit == 2:
                delta = 2 * power
            else:
                delta = -power
            n_idx = idx + delta
            neighbors[idx, neighbor_idx] = n_idx
            neighbor_idx += 1

    # Helper to convert index to word
    def idx_to_word(idx):
        w = [0] * n
        temp = idx
        for i in range(n):
            w[i] = temp % q
            temp //= q
        return w

    def luby_greedy(seed):
        rng = np.random.default_rng(seed)
        is_covered = np.zeros(total_points, dtype=bool)
        selected = []

        while not np.all(is_covered):
            uncovered_counts = np.sum(~is_covered[neighbors], axis=1)
            max_count = np.max(uncovered_counts)
            if max_count == 0:
                break

            # Luby weight^2: pick with probability proportional to count^2
            weights = uncovered_counts.astype(np.float64)
            weights[weights == 0] = 0
            weights_sq = weights * weights
            total_w = weights_sq.sum()
            if total_w == 0:
                break
            probs = weights_sq / total_w
            best_idx = rng.choice(total_points, p=probs)

            selected.append(best_idx)
            is_covered[neighbors[best_idx]] = True

        return selected

    def prune(selected_indices):
        if not selected_indices:
            return []

        selected_set = set(selected_indices)
        coverage_count = np.zeros(total_points, dtype=np.int32)

        for idx in selected_set:
            coverage_count[neighbors[idx]] += 1

        changed = True
        while changed:
            changed = False
            # Vectorized: find all redundant codewords at once
            # A codeword idx is redundant if all points in its ball have coverage > 1
            # Check for each selected idx
            redundant_mask = np.zeros(len(selected_set), dtype=bool)
            sel_list = list(selected_set)
            for i, idx in enumerate(sel_list):
                if np.all(coverage_count[neighbors[idx]] > 1):
                    redundant_mask[i] = True
                    changed = True

            if changed:
                for i in range(len(sel_list)):
                    if redundant_mask[i]:
                        rem_idx = sel_list[i]
                        coverage_count[neighbors[rem_idx]] -= 1
                        selected_set.discard(rem_idx)

        return list(selected_set)

    def local_search_optimize(selected_indices, deadline):
        selected_list = list(selected_indices)
        selected_list = prune(selected_list)
        selected_set = set(selected_list)

        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[neighbors[idx]] += 1

        best_len = len(selected_list)
        best_code = selected_list[:]

        rng = np.random.default_rng(42)

        while time.time() < deadline:
            if not selected_list:
                break

            # Try multiple random swaps per iteration
            for _ in range(10):
                if time.time() > deadline:
                    break

                c = selected_list[rng.integers(len(selected_list))]
                c_prime = neighbors[c][rng.integers(1, ball_size)]

                if c_prime in selected_set:
                    continue

                # Tentatively swap c -> c_prime
                coverage_count[neighbors[c]] -= 1
                coverage_count[neighbors[c_prime]] += 1

                if np.any(coverage_count[neighbors[c]] == 0):
                    # Revert
                    coverage_count[neighbors[c_prime]] -= 1
                    coverage_count[neighbors[c]] += 1
                    continue
                else:
                    # Accept swap
                    pos = selected_list.index(c)
                    selected_list[pos] = c_prime
                    selected_set.discard(c)
                    selected_set.add(c_prime)

                    # Try to prune
                    pruned = prune(selected_list)
                    if len(pruned) < len(selected_list):
                        selected_list = pruned
                        selected_set = set(selected_list)
                        coverage_count = np.zeros(total_points, dtype=np.int32)
                        for idx in selected_list:
                            coverage_count[neighbors[idx]] += 1

                        if len(selected_list) < best_len:
                            best_len = len(selected_list)
                            best_code = selected_list[:]

        return best_code

    def two_opt_search(selected_indices, deadline):
        """More aggressive search: try removing and re-adding codewords."""
        selected_list = list(selected_indices)
        selected_list = prune(selected_list)
        selected_set = set(selected_list)

        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[neighbors[idx]] += 1

        best_len = len(selected_list)
        best_code = selected_list[:]

        rng = np.random.default_rng(123)

        while time.time() < deadline:
            if not selected_list:
                break

            # Try removing a random codeword and see if we can recover coverage
            c = selected_list[rng.integers(len(selected_list))]
            coverage_count[neighbors[c]] -= 1

            # Find uncovered points
            uncovered = np.where(coverage_count == 0)[0]

            if len(uncovered) == 0:
                # We can remove c!
                pos = selected_list.index(c)
                selected_list.pop(pos)
                selected_set.discard(c)
                if len(selected_list) < best_len:
                    best_len = len(selected_list)
                    best_code = selected_list[:]
            else:
                # Try to cover uncovered points with new codewords
                # For each uncovered point, try adding a codeword that covers it
                covered_new = 0
                for unc_idx in uncovered:
                    # Find a codeword that covers this point and minimizes overlap
                    best_new = -1
                    best_gain = -1
                    for candidate in neighbors[unc_idx]:
                        if candidate in selected_set:
                            continue
                        gain = np.sum(coverage_count[neighbors[candidate]] == 0)
                        if gain > best_gain:
                            best_gain = gain
                            best_new = candidate
                    if best_new >= 0:
                        selected_list.append(best_new)
                        selected_set.add(best_new)
                        coverage_count[neighbors[best_new]] += 1
                        covered_new += 1
                        if covered_new >= len(uncovered):
                            break

                if covered_new < len(uncovered):
                    # Revert: remove the newly added codewords
                    for _ in range(covered_new):
                        c2 = selected_list.pop()
                        selected_set.discard(c2)
                        coverage_count[neighbors[c2]] -= 1

                # Restore c
                coverage_count[neighbors[c]] += 1
                if c not in selected_set:
                    selected_list.append(c)
                    selected_set.add(c)

        return best_code

    def simulated_annealing(selected_indices, deadline):
        """SA-based local search."""
        selected_list = list(selected_indices)
        selected_list = prune(selected_list)
        selected_set = set(selected_list)

        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[neighbors[idx]] += 1

        best_len = len(selected_list)
        best_code = selected_list[:]

        rng = np.random.default_rng(777)
        temp = 5.0
        cooling = 0.995

        while time.time() < deadline and temp > 0.1:
            if not selected_list:
                break

            for _ in range(20):
                if time.time() > deadline:
                    break

                # Random move: swap, add, or remove
                move_type = rng.integers(3)

                if move_type == 0:  # Swap
                    if len(selected_list) < 2:
                        continue
                    c = selected_list[rng.integers(len(selected_list))]
                    c_prime = neighbors[c][rng.integers(1, ball_size)]
                    if c_prime in selected_set:
                        continue

                    coverage_count[neighbors[c]] -= 1
                    coverage_count[neighbors[c_prime]] += 1

                    if np.any(coverage_count[neighbors[c]] == 0):
                        coverage_count[neighbors[c_prime]] -= 1
                        coverage_count[neighbors[c]] += 1
                        continue
                    else:
                        pos = selected_list.index(c)
                        selected_list[pos] = c_prime
                        selected_set.discard(c)
                        selected_set.add(c_prime)

                        pruned = prune(selected_list)
                        if len(pruned) < len(selected_list):
                            selected_list = pruned
                            selected_set = set(selected_list)
                            coverage_count = np.zeros(total_points, dtype=np.int32)
                            for idx in selected_list:
                                coverage_count[neighbors[idx]] += 1

                        if len(selected_list) < best_len:
                            best_len = len(selected_list)
                            best_code = selected_list[:]

                elif move_type == 1:  # Add
                    c = rng.integers(total_points)
                    if c in selected_set:
                        continue
                    selected_list.append(c)
                    selected_set.add(c)
                    coverage_count[neighbors[c]] += 1

                    pruned = prune(selected_list)
                    if len(pruned) < len(selected_list):
                        selected_list = pruned
                        selected_set = set(selected_list)
                        coverage_count = np.zeros(total_points, dtype=np.int32)
                        for idx in selected_list:
                            coverage_count[neighbors[idx]] += 1

                    if len(selected_list) < best_len:
                        best_len = len(selected_list)
                        best_code = selected_list[:]

                else:  # Remove
                    if len(selected_list) < 2:
                        continue
                    c = selected_list[rng.integers(len(selected_list))]
                    coverage_count[neighbors[c]] -= 1

                    if np.any(coverage_count[neighbors[c]] == 0):
                        coverage_count[neighbors[c]] += 1
                        continue
                    else:
                        pos = selected_list.index(c)
                        selected_list.pop(pos)
                        selected_set.discard(c)

                        if len(selected_list) < best_len:
                            best_len = len(selected_list)
                            best_code = selected_list[:]

            temp *= cooling

        return best_code

    start_time = time.time()
    deadline = start_time + 118.0

    # Run greedy instances
    best_code = None
    best_len = float('inf')

    # Run multiple greedy instances with different seeds
    for seed in range(30):
        if time.time() > deadline - 70:
            break
        selected = luby_greedy(seed)
        selected = prune(selected)
        if len(selected) < best_len:
            best_len = len(selected)
            best_code = selected

    if best_code is None:
        best_code = luby_greedy(0)
        best_code = prune(best_code)
        best_len = len(best_code)

    # Local search phases
    remaining = deadline - time.time()
    phase1_deadline = start_time + remaining * 0.4
    phase2_deadline = start_time + remaining * 0.7
    phase3_deadline = deadline

    # Phase 1: Standard local search
    best_code = local_search_optimize(best_code, phase1_deadline)
    best_len = len(best_code)

    # Phase 2: Two-opt search
    best_code = two_opt_search(best_code, phase2_deadline)
    best_len = len(best_code)

    # Phase 3: Simulated annealing
    best_code = simulated_annealing(best_code, phase3_deadline)
    best_len = len(best_code)

    # Final prune
    best_code = prune(best_code)

    # Convert to words
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))

    return result
