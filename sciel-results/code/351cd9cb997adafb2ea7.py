"""Combined program: algebraic constructive initialization + aggressive local search."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    if q == 3 and n == 7 and radius == 1:
        return _solve_ternary_7_1(start_time, deadline)

    return _direct_sum_construct(q, n, radius)


def _direct_sum_construct(q, n, radius):
    if n < 4:
        if n == 0:
            return [[]]
        if n == 1:
            return [[i] for i in range(q)]
        if n == 2:
            return [[a, b] for a in range(q) for b in range(q)]
        if n == 3:
            return [[a, b, (a + b) % q] for a in range(q) for b in range(q)]

    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        result = []
        for h in tetracode:
            for tail in product(range(3), repeat=n - 4):
                result.append(h + list(tail))
        return result
    else:
        return [list(w) for w in product(range(q), repeat=n)]


def _solve_ternary_7_1(start_time, deadline):
    q, n, radius = 3, 7, 1
    N = q ** n  # 2187
    ball_size = 1 + n * (q - 1)  # 14
    # Adjust deadline to be safe within 120s limit (evaluator kills at 150s, but task says 120s)
    # We are passed a deadline, but we should enforce a stricter one to be safe.
    # The caller passes deadline = start_time + 145. We will override with start_time + 110.
    deadline = start_time + 110

    # Precompute all words
    all_words = [list(w) for w in product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}

    # Precompute neighbors (ball contents) as numpy arrays
    neighbors = np.empty((N, ball_size), dtype=np.int32)
    for i, w in enumerate(all_words):
        w_list = list(w)
        neighbors[i, 0] = i
        pos = 1
        for j in range(n):
            orig_val = w_list[j]
            for val in range(q):
                if val != orig_val:
                    w_list[j] = val
                    neighbors[i, pos] = word_to_idx[tuple(w_list)]
                    pos += 1
            w_list[j] = orig_val

    # Precompute coverers: coverers[point] = list of codewords that cover point
    coverers = [[] for _ in range(N)]
    for cw_idx in range(N):
        for pt in neighbors[cw_idx]:
            coverers[pt].append(cw_idx)

    # Convert coverers to numpy arrays for faster intersection
    coverer_arrays = [np.array(c, dtype=np.int32) for c in coverers]

    # 1. Initial Construction: Fast Greedy Set Cover with bitsets
    ball_bitsets = [0] * N
    for i in range(N):
        mask = 0
        for j in neighbors[i]:
            mask |= (1 << j)
        ball_bitsets[i] = mask

    all_mask = (1 << N) - 1
    current_cover_mask = 0
    codewords = []
    is_chosen = np.zeros(N, dtype=bool)

    # Greedy Set Cover
    while current_cover_mask != all_mask:
        if time.time() > deadline - 100:
            break

        best_i = -1
        best_count = 0

        for i in range(N):
            if is_chosen[i]:
                continue
            new_bits = ball_bitsets[i] & ~current_cover_mask
            if new_bits:
                count = new_bits.bit_count()
                if count > best_count:
                    best_count = count
                    best_i = i

        if best_i == -1:
            break

        codewords.append(best_i)
        is_chosen[best_i] = True
        current_cover_mask |= ball_bitsets[best_i]

    # Ensure validity
    if current_cover_mask != all_mask:
        while current_cover_mask != all_mask:
            i = random.randint(0, N - 1)
            if not is_chosen[i]:
                codewords.append(i)
                is_chosen[i] = True
            current_cover_mask |= ball_bitsets[i]
            if time.time() > deadline - 10:
                break

    # 2. Local Search Setup
    cover_count = np.zeros(N, dtype=np.int16)
    for cw in codewords:
        cover_count[neighbors[cw]] += 1

    best_codewords = list(codewords)
    best_size = len(codewords)
    current_codeword_set = set(codewords)

    # Helper: Try to remove redundant codewords
    def try_remove_redundant(codewords, cover_count, time_limit):
        new_codewords = []
        removed = 0
        for cw in codewords:
            if time.time() > time_limit:
                # If time runs out, keep the rest to maintain validity (though we might miss removals)
                new_codewords.extend(codewords[codewords.index(cw):]) # This is O(N^2) if done badly, but only once at timeout
                break
            b = neighbors[cw]
            if np.min(cover_count[b]) > 1:
                cover_count[b] -= 1
                removed += 1
            else:
                new_codewords.append(cw)
        return new_codewords, removed

    # Initial redundancy removal
    codewords, _ = try_remove_redundant(codewords, cover_count, deadline - 30)
    current_codeword_set = set(codewords)
    if len(codewords) < best_size:
        best_size = len(codewords)
        best_codewords = list(codewords)

    # Local Search Loop
    random.seed(42)
    no_improvement_count = 0
    max_no_improvement = 100

    while time.time() < deadline - 3:
        improved = False

        # 1. Try Removals
        new_codewords, removed = try_remove_redundant(codewords, cover_count, deadline - 5)
        if removed > 0:
            codewords = new_codewords
            current_codeword_set = set(codewords)
            improved = True
            no_improvement_count = 0
            if len(codewords) < best_size:
                best_size = len(codewords)
                best_codewords = list(codewords)
            continue

        # 2. Try Swaps with Critical Word Heuristic
        if codewords:
            # Pick a random codeword to try swapping
            idx = random.randint(0, len(codewords) - 1)
            old_cw = codewords[idx]

            # Find critical points: points in ball(old_cw) with count == 1
            b_old = neighbors[old_cw]
            critical_points = b_old[cover_count[b_old] == 1]

            if len(critical_points) == 0:
                # Should have been removed by redundancy check
                cover_count[b_old] -= 1
                codewords.pop(idx)
                improved = True
                no_improvement_count = 0
                if len(codewords) < best_size:
                    best_size = len(codewords)
                    best_codewords = list(codewords)
                continue

            # Find candidates that cover ALL critical points using numpy intersection
            # Start with the coverers of the critical point with fewest coverers
            min_len = float('inf')
            min_pt = critical_points[0]
            for pt in critical_points:
                cl = len(coverer_arrays[pt])
                if cl < min_len:
                    min_len = cl
                    min_pt = pt

            candidates = coverer_arrays[min_pt]
            for p in critical_points:
                if p == min_pt:
                    continue
                candidates = np.intersect1d(candidates, coverer_arrays[p])
            
            if len(candidates) == 0:
                continue

            # Remove old_cw and existing codewords from candidates
            # Convert to list for filtering, or use boolean mask
            # Since len(candidates) is usually small, list comprehension is fine
            candidates_list = [c for c in candidates if c != old_cw and c not in current_codeword_set]

            if candidates_list:
                if len(candidates_list) > 30:
                    random.shuffle(candidates_list)
                    candidates_list = candidates_list[:30]

                best_new_cw = -1
                min_cost = float('inf')

                # Vectorized cost: for each candidate, count how many points in its ball have cover_count > 1
                for new_cw in candidates_list:
                    b_new = neighbors[new_cw]
                    # Cost: number of points with cover_count > 1 (overlap with existing coverage)
                    # Lower cost = less redundancy = better
                    cost = np.sum(cover_count[b_new] > 1)
                    if cost < min_cost:
                        min_cost = cost
                        best_new_cw = new_cw

                if best_new_cw != -1:
                    # Perform swap
                    cover_count[b_old] -= 1
                    cover_count[neighbors[best_new_cw]] += 1
                    codewords[idx] = best_new_cw
                    current_codeword_set.remove(old_cw)
                    current_codeword_set.add(best_new_cw)
                    improved = True
                    no_improvement_count = 0
                    # Try to remove redundancies after swap
                    new_codewords, _ = try_remove_redundant(codewords, cover_count, deadline - 5)
                    if len(new_codewords) < len(codewords):
                        codewords = new_codewords
                        current_codeword_set = set(codewords)
                    if len(codewords) < best_size:
                        best_size = len(codewords)
                        best_codewords = list(codewords)
                    continue

        if not improved:
            no_improvement_count += 1
            if no_improvement_count > max_no_improvement:
                break

    # Return the best result found
    result = best_codewords
    return [all_words[i] for i in result]
# EVOLVE-BLOCK-END
