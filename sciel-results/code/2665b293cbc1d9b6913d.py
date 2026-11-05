"""Football pool / ternary covering code K_3(8,1).
Strategy: numpy-accelerated coverage counting + multi-restart local search
with neighbor-restricted moves. Start from direct-sum construction, prune,
then explore via random walks in codeword space.
"""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill

    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    total_points = q ** n          # 6561
    ball_size = 1 + n * (q - 1)    # 17

    # Pre-compute powers for fast index arithmetic
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Pre-compute neighbors (ball of radius 1) for each point
    neighbors = [None] * total_points
    for idx in range(total_points):
        temp_idx = idx
        word = [0] * n
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q

        nbrs = [idx]
        for pos in range(n):
            orig_digit = word[pos]
            power = powers[pos]

            # +1 mod q
            new_digit = (orig_digit + 1) % q
            if new_digit == 0:
                delta = -2 * power
            else:
                delta = power
            nbrs.append(idx + delta)

            # -1 mod q
            new_digit = (orig_digit - 1) % q
            if new_digit == 2:
                delta = 2 * power
            else:
                delta = -power
            nbrs.append(idx + delta)

        neighbors[idx] = nbrs

    # Vectorized balls matrix (6561, 17)
    balls = np.array(neighbors, dtype=np.int32)

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w

    # Build initial code: direct sum of tetracode with all 3^4 tails = 9*81 = 729 words
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_indices = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=4):
            word = h + list(tail)
            idx = 0
            for v in word:
                idx = idx * 3 + v
            initial_indices.append(idx)

    def prune(selected_indices, coverage_count=None):
        """Remove redundant codewords. Returns pruned list and coverage count."""
        if not selected_indices:
            return selected_indices, coverage_count
        selected_list = list(selected_indices)
        if coverage_count is None:
            coverage_count = np.zeros(total_points, dtype=np.int32)
            for idx in selected_list:
                coverage_count[balls[idx]] += 1
        changed = True
        while changed:
            changed = False
            for i in range(len(selected_list)):
                idx = selected_list[i]
                if np.all(coverage_count[balls[idx]] > 1):
                    coverage_count[balls[idx]] -= 1
                    selected_list.pop(i)
                    changed = True
                    break
        return selected_list, coverage_count

    def rebuild_coverage(selected_list):
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[balls[idx]] += 1
        return coverage_count

    def local_search(selected_indices, rng, max_steps=50000):
        """Local search with neighbor-restricted moves."""
        selected_list = list(selected_indices)
        coverage_count = rebuild_coverage(selected_list)
        selected_set = set(selected_list)

        best_len = len(selected_list)
        best_code = selected_list[:]

        # Initial prune
        selected_list, coverage_count = prune(selected_list, coverage_count)
        selected_set = set(selected_list)
        if len(selected_list) < best_len:
            best_len = len(selected_list)
            best_code = selected_list[:]

        steps = 0
        while steps < max_steps and time.time() < deadline:
            # Deterministic removal sweep
            changed = False
            for i in range(len(selected_list)):
                idx = selected_list[i]
                if np.all(coverage_count[balls[idx]] > 1):
                    coverage_count[balls[idx]] -= 1
                    selected_list.pop(i)
                    selected_set.discard(idx)
                    changed = True
                    if len(selected_list) < best_len:
                        best_len = len(selected_list)
                        best_code = selected_list[:]
                    break

            if not changed and time.time() > deadline:
                break

            n_sel = len(selected_list)
            if n_sel == 0:
                break

            # Try a neighbor-restricted move
            progress = False
            for _ in range(20):
                if time.time() > deadline:
                    break
                i = rng.randrange(n_sel)
                c = selected_list[i]
                nbrs = list(neighbors[c])
                rng.shuffle(nbrs)
                for c_prime in nbrs:
                    if c_prime == c or c_prime in selected_set:
                        continue
                    # Tentatively move
                    coverage_count[balls[c]] -= 1
                    if np.any(coverage_count[balls[c]] == 0):
                        coverage_count[balls[c]] += 1
                        continue
                    # Valid move
                    coverage_count[balls[c_prime]] += 1
                    selected_list[i] = c_prime
                    selected_set.discard(c)
                    selected_set.add(c_prime)
                    progress = True
                    break
                if progress:
                    break

            steps += 1

            # Periodic re-prune
            if steps % 100 == 0:
                pruned, coverage_count = prune(selected_list, coverage_count)
                selected_set = set(pruned)
                selected_list = pruned
                if len(selected_list) < best_len:
                    best_len = len(selected_list)
                    best_code = selected_list[:]

        return best_code

    # Multi-restart local search
    best_overall = None
    best_len_overall = float('inf')

    # Start from the initial construction
    initial_pruned, _ = prune(initial_indices)
    if len(initial_pruned) < best_len_overall:
        best_len_overall = len(initial_pruned)
        best_overall = initial_pruned[:]

    # Run multiple local search phases with different seeds
    for seed in [42, 123, 456, 789, 101, 202, 303, 404]:
        if time.time() > deadline - 5:
            break
        rng = random.Random(seed)
        # Start from the pruned initial code
        start_list = initial_pruned[:]
        # Shuffle for diversity
        rng.shuffle(start_list)
        result = local_search(start_list, rng, max_steps=30000)
        if len(result) < best_len_overall:
            best_len_overall = len(result)
            best_overall = result[:]

    # Final polish on the best solution
    if best_overall and time.time() < deadline - 3:
        rng = random.Random(999)
        final = local_search(best_overall, rng, max_steps=20000)
        if len(final) < best_len_overall:
            best_overall = final

    return [idx_to_word(idx) for idx in best_overall]
# EVOLVE-BLOCK-END
