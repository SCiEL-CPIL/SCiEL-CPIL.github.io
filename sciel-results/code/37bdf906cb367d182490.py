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
        rng = np.random.default_rng(seed) if not deterministic else None
        is_covered = np.zeros(total_points, dtype=bool)
        selected = []
        gain = np.full(total_points, ball_size, dtype=np.int32)

        while not np.all(is_covered):
            max_gain = np.max(gain)
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
                    gain[neighbors[p]] -= 1

        return selected

    def rebuild_coverage(selected_list):
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[balls[idx]] += 1
        return coverage_count

    def prune(selected_list, coverage_count=None):
        if not selected_list:
            return selected_list
        selected_list = list(selected_list)
        if coverage_count is None:
            coverage_count = rebuild_coverage(selected_list)
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
        return selected_list

    def local_search_walk(selected_indices, deadline):
        selected_list = list(selected_indices)
        selected_list = prune(selected_list)
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

            # Neighbor-restricted swap: move a random codeword within its own ball
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
        selected_list = list(selected_indices)
        coverage_count = rebuild_coverage(selected_list)
        selected_set = set(selected_list)
        best_len = len(selected_list)
        best_code = selected_list[:]
        rng = random.Random(7)

        for _ in range(200):
            if time.time() > deadline:
                break
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

            if not changed:
                # Try a single neighbor swap
                n_sel = len(selected_list)
                if n_sel == 0:
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

        return best_code

    start_time = time.time()
    deadline = start_time + 115.0

    # Run greedy instances and pick the best after pruning
    best_code = None
    best_len = float('inf')

    num_greedy = 8
    for seed in range(num_greedy):
        if time.time() > deadline - 60:
            break
        selected = greedy_set_cover(seed)
        selected = prune(selected)
        if len(selected) < best_len:
            best_len = len(selected)
            best_code = selected

    if best_code is None:
        best_code = greedy_set_cover(0)
        best_code = prune(best_code)
        best_len = len(best_code)

    # Local search on the best code
    best_code = local_search_walk(best_code, deadline - 15.0)

    # Final polish
    best_code = polish(best_code, deadline - 2.0)

    # Convert to words
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))

    return result
