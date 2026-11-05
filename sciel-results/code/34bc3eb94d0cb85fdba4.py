import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
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
            new_digit = (orig_digit + 1) % q
            delta = -2 * power if new_digit == 0 else power
            nbrs.append(idx + delta)
            new_digit = (orig_digit - 1) % q
            delta = 2 * power if new_digit == 2 else -power
            nbrs.append(idx + delta)
        neighbors[idx] = nbrs

    balls = np.array(neighbors, dtype=np.int32)  # (6561, 17)

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w

    def greedy_set_cover(seed):
        """Incremental-gain greedy: O(ball_size) per selection instead of O(N*ball_size)."""
        rng = np.random.default_rng(seed)
        is_covered = np.zeros(total_points, dtype=bool)
        selected = []
        gain = np.full(total_points, ball_size, dtype=np.int32)

        while not np.all(is_covered):
            max_gain = int(np.max(gain))
            candidates = np.where(gain == max_gain)[0]
            if len(candidates) > 1:
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
            return list(selected_list)
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

    def local_search_optimize(selected_indices, deadline):
        """Neighbor swaps + 2-opt escape under deadline."""
        selected_list = prune(list(selected_indices))
        coverage_count = rebuild_coverage(selected_list)
        selected_set = set(selected_list)
        best_len = len(selected_list)
        best_code = selected_list[:]
        rng = random.Random(42)

        while time.time() < deadline:
            # 1. Deterministic removal sweep to fixed point
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

            # 2. Neighbor-restricted swap (vectorized validity check)
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
                    coverage_count[balls[c]] -= 1
                    if np.any(coverage_count[balls[c]] == 0):
                        coverage_count[balls[c]] += 1
                        continue
                    # valid swap
                    coverage_count[balls[c_prime]] += 1
                    selected_list[i] = c_prime
                    selected_set.discard(c)
                    selected_set.add(c_prime)
                    progress = True
                    break
                if progress:
                    break

            # 3. 2-opt escape: remove two, add one
            if time.time() < deadline and len(selected_list) >= 2:
                n_sel = len(selected_list)
                for _ in range(20):
                    if time.time() > deadline:
                        break
                    i = rng.randrange(n_sel)
                    j = rng.randrange(n_sel)
                    if i == j:
                        continue
                    a = selected_list[i]
                    b = selected_list[j]
                    coverage_count[balls[a]] -= 1
                    coverage_count[balls[b]] -= 1
                    # exposed points: those whose count dropped to 0
                    exposed_a = balls[a][coverage_count[balls[a]] == 0]
                    exposed_b = balls[b][coverage_count[balls[b]] == 0]
                    if len(exposed_a) > 0 or len(exposed_b) > 0:
                        # need a single codeword covering all exposed points
                        # Try a few random candidates from the neighborhood of exposed points
                        found = False
                        if len(exposed_a) > 0:
                            ea = int(exposed_a[0])
                            for cand in neighbors[ea]:
                                if cand == a or cand == b or cand in selected_set:
                                    continue
                                if np.all(coverage_count[balls[cand]] > 0) or True:
                                    # check cand covers all exposed
                                    all_exposed = list(exposed_a) + list(exposed_b)
                                    ok = True
                                    for ep in all_exposed:
                                        if ep not in neighbors[cand]:
                                            ok = False
                                            break
                                    if ok:
                                        coverage_count[balls[cand]] += 1
                                        # remove a and b
                                        selected_list.remove(a)
                                        selected_list.remove(b)
                                        selected_set.discard(a)
                                        selected_set.discard(b)
                                        selected_list.append(cand)
                                        selected_set.add(cand)
                                        progress = True
                                        found = True
                                        break
                        if not found:
                            # revert
                            coverage_count[balls[a]] += 1
                            coverage_count[balls[b]] += 1
                    else:
                        # both a and b were redundant (shouldn't happen after prune, but safety)
                        selected_list.remove(a)
                        selected_list.remove(b)
                        selected_set.discard(a)
                        selected_set.discard(b)
                        progress = True
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
        """High-precision final pass: deterministic sweeps + aggressive neighbor swaps."""
        selected_list = list(selected_indices)
        coverage_count = rebuild_coverage(selected_list)
        selected_set = set(selected_list)
        best_len = len(selected_list)
        best_code = selected_list[:]
        rng = random.Random(7)

        for _ in range(500):
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

            if not selected_list:
                break

            # Try neighbor swaps
            n_sel = len(selected_list)
            swapped = False
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
                    coverage_count[balls[c]] -= 1
                    if np.any(coverage_count[balls[c]] == 0):
                        coverage_count[balls[c]] += 1
                        continue
                    coverage_count[balls[c_prime]] += 1
                    selected_list[i] = c_prime
                    selected_set.discard(c)
                    selected_set.add(c_prime)
                    swapped = True
                    break
                if swapped:
                    break

            if swapped:
                # prune after swap
                pruned = prune(selected_list, coverage_count)
                if len(pruned) < len(selected_list):
                    coverage_count = rebuild_coverage(pruned)
                    selected_list = pruned
                    selected_set = set(selected_list)
                if len(selected_list) < best_len:
                    best_len = len(selected_list)
                    best_code = selected_list[:]

        return best_code

    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    # Phase 1: Multiple greedy initializations
    best_code = None
    best_len = float('inf')
    for seed in range(10):
        if time.time() > deadline - 60:
            break
        selected = greedy_set_cover(seed)
        selected = prune(selected)
        if len(selected) < best_len:
            best_len = len(selected)
            best_code = selected[:]

    if best_code is None:
        best_code = prune(greedy_set_cover(0))
        best_len = len(best_code)

    # Phase 2: Local search with 2-opt escape
    local_deadline = min(deadline - 25, start_time + 90.0)
    best_code = local_search_optimize(best_code, local_deadline)

    # Phase 3: High-precision polish
    best_code = polish(best_code, deadline)

    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))
    return result
