"""
Optimized Ternary Covering Code K_3(8,1) Solver.
Combines fast incremental-gain greedy with neighbor-restricted swaps and aggressive pruning.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            return list(itertools.product(range(q), repeat=n))

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

            # +1
            new_digit = (orig_digit + 1) % q
            if new_digit == 0:
                delta = -2 * power
            else:
                delta = power
            nbrs.append(idx + delta)

            # -1
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

    def greedy_set_cover(seed, deterministic=False):
        """
        Incremental-gain greedy set cover.
        Maintains a gain array that tracks how many new points each candidate covers.
        """
        rng = np.random.default_rng(seed) if not deterministic else None
        is_covered = np.zeros(total_points, dtype=bool)
        selected = []
        # Initialize gain: each candidate covers ball_size points initially
        gain = np.full(total_points, ball_size, dtype=np.int32)

        while not np.all(is_covered):
            max_gain = np.max(gain)
            if max_gain == 0:
                break
            candidates = np.where(gain == max_gain)[0]
            if len(candidates) > 1 and not deterministic:
                best_idx = int(rng.choice(candidates))
            else:
                best_idx = int(candidates[0])

            selected.append(best_idx)
            new_points = neighbors[best_idx]
            for p in new_points:
                if not is_covered[p]:
                    is_covered[p] = True
                    # Decrease gain for all candidates that cover p
                    gain[balls[p]] -= 1

        return selected

    def rebuild_coverage(selected_list):
        """Compute coverage count for each point given a list of selected codewords."""
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[balls[idx]] += 1
        return coverage_count

    def prune(selected_list, coverage_count=None):
        """
        Remove redundant codewords. A codeword is redundant if every point in its ball
        is covered by at least one other selected codeword.
        """
        if not selected_list:
            return selected_list
        selected_list = list(selected_list)
        if coverage_count is None:
            coverage_count = rebuild_coverage(selected_list)
        else:
            coverage_count = coverage_count.copy()

        changed = True
        while changed:
            changed = False
            for i in range(len(selected_list)):
                idx = selected_list[i]
                # Check if any point in this ball has coverage exactly 1 (only this codeword covers it)
                if np.all(coverage_count[balls[idx]] > 1):
                    # Redundant: remove it
                    coverage_count[balls[idx]] -= 1
                    selected_list.pop(i)
                    changed = True
                    break

        return selected_list

    def local_search_walk(selected_indices, deadline):
        """
        Neighbor-restricted swaps + deterministic removal sweeps.
        Tries to move codewords to nearby positions that maintain coverage while allowing pruning.
        """
        selected_list = list(selected_indices)
        coverage_count = rebuild_coverage(selected_list)
        selected_set = set(selected_list)

        best_len = len(selected_list)
        best_code = selected_list[:]
        rng = random.Random(42)

        while time.time() < deadline:
            # Deterministic removal sweep until fixed point
            removed_any = True
            while removed_any and time.time() < deadline:
                removed_any = False
                for i in range(len(selected_list)):
                    idx = selected_list[i]
                    if np.all(coverage_count[balls[idx]] > 1):
                        coverage_count[balls[idx]] -= 1
                        selected_list.pop(i)
                        selected_set.discard(idx)
                        removed_any = True
                        if len(selected_list) < best_len:
                            best_len = len(selected_list)
                            best_code = selected_list[:]
                        break

            if time.time() > deadline:
                break

            n_sel = len(selected_list)
            if n_sel == 0:
                break

            # Try neighbor-restricted swaps
            progress = False
            for _ in range(40):
                if time.time() > deadline:
                    break
                i = rng.randrange(n_sel)
                c = selected_list[i]
                nbrs = list(neighbors[c])
                rng.shuffle(nbrs)
                for c_prime in nbrs:
                    if c_prime == c or c_prime in selected_set:
                        continue
                    # Tentatively swap
                    coverage_count[balls[c]] -= 1
                    coverage_count[balls[c_prime]] += 1
                    if np.any(coverage_count[balls[c]] == 0):
                        # Invalid, revert
                        coverage_count[balls[c_prime]] -= 1
                        coverage_count[balls[c]] += 1
                        continue
                    # Valid swap
                    selected_list[i] = c_prime
                    selected_set.discard(c)
                    selected_set.add(c_prime)
                    progress = True
                    break
                if progress:
                    break

            if progress:
                # After a swap, try to prune
                pruned = prune(selected_list, coverage_count)
                if len(pruned) < len(selected_list):
                    coverage_count = rebuild_coverage(pruned)
                    selected_list = pruned
                    selected_set = set(selected_list)
                if len(selected_list) < best_len:
                    best_len = len(selected_list)
                    best_code = selected_list[:]

        return best_code

    def polish(selected_indices, deadline):
        """
        High-precision polish pass: multiple rounds of deterministic sweeps and neighbor swaps
        with different random seeds to escape local minima.
        """
        selected_list = list(selected_indices)
        coverage_count = rebuild_coverage(selected_list)
        selected_set = set(selected_list)
        best_len = len(selected_list)
        best_code = selected_list[:]

        # Multiple rounds with different random seeds
        for round_num in range(5):
            if time.time() > deadline:
                break
            rng = random.Random(7 + round_num * 13)

            # Deterministic removal sweep
            changed = True
            while changed and time.time() < deadline:
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

            if time.time() > deadline:
                break

            # Neighbor-restricted swaps
            n_sel = len(selected_list)
            if n_sel == 0:
                break

            for _ in range(60):
                if time.time() > deadline:
                    break
                i = rng.randrange(n_sel)
                c = selected_list[i]
                nbrs = list(neighbors[c])
                rng.shuffle(nbrs)
                for c_prime in nbrs:
                    if c_prime == c or c_prime in selected_set:
                        continue
                    coverage_count[balls[c]] -= 1
                    coverage_count[balls[c_prime]] += 1
                    if np.any(coverage_count[balls[c]] == 0):
                        coverage_count[balls[c_prime]] -= 1
                        coverage_count[balls[c]] += 1
                        continue
                    selected_list[i] = c_prime
                    selected_set.discard(c)
                    selected_set.add(c_prime)
                    break

            # Prune after swaps
            pruned = prune(selected_list, coverage_count)
            if len(pruned) < len(selected_list):
                coverage_count = rebuild_coverage(pruned)
                selected_list = pruned
                selected_set = set(selected_list)
            if len(selected_list) < best_len:
                best_len = len(selected_list)
                best_code = selected_list[:]

        return best_code

    # Main Loop
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    best_code = None
    best_len = float('inf')

    # Phase 1: Run multiple randomized greedy passes to find good starting codes
    iteration = 0
    while time.time() < deadline:
        selected = greedy_set_cover(iteration)
        selected = prune(selected)

        if len(selected) < best_len:
            best_len = len(selected)
            best_code = selected

        if best_len <= 486:
            break

        iteration += 1

    if best_code is None:
        best_code = greedy_set_cover(0)
        best_code = prune(best_code)
        best_len = len(best_code)

    # Phase 2: Local search on the best code found
    if time.time() < deadline:
        improved = local_search_walk(best_code, deadline)
        if len(improved) < best_len:
            best_len = len(improved)
            best_code = improved

    # Phase 3: Polish pass
    if time.time() < deadline:
        polished = polish(best_code, deadline)
        if len(polished) < best_len:
            best_len = len(polished)
            best_code = polished

    # Convert indices to words
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))

    return result

# EVOLVE-BLOCK-END
