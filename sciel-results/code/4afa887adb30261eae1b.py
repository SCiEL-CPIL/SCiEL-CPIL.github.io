import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)

    # Fallback for other parameters
    if q ** n <= 10000:
        all_words = [list(c) for c in itertools.product(range(q), repeat=n)]
        uncovered = set(range(len(all_words)))
        codewords = []

        def hamming_dist(a, b):
            return sum(1 for x, y in zip(a, b) if x != y)

        while uncovered:
            if time.time() > deadline - 5.0:
                break
            best_idx = -1
            best_count = -1
            for i in range(len(all_words)):
                count = sum(1 for j in uncovered if hamming_dist(all_words[i], all_words[j]) <= radius)
                if count > best_count:
                    best_count = count
                    best_idx = i
            if best_idx == -1:
                break
            codewords.append(all_words[best_idx])
            uncovered = {j for j in uncovered if hamming_dist(all_words[best_idx], all_words[j]) > radius}
        return codewords
    else:
        return [list(c) for c in itertools.product(range(q), repeat=n)]


def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17

    # Precompute base-3 weights
    weights = [3 ** (n - 1 - i) for i in range(n)]

    # Precompute balls: balls[w, 0..16] are the 17 words within Hamming distance 1 of w
    balls = np.zeros((total, ball_size), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1

    # ball_membership[w, c] = True iff c covers w (symmetric with balls)
    ball_membership = np.zeros((total, total), dtype=np.int8)
    for w in range(total):
        ball_membership[w, balls[w]] = 1

    def idx_to_digits(idx):
        digits = [0] * n
        for i in range(n - 1, -1, -1):
            digits[i] = idx % q
            idx //= q
        return digits

    def digits_to_idx(digits):
        idx = 0
        for i in range(n):
            idx = idx * q + digits[i]
        return idx

    # ---------------- Partition-based recursive construction ----------------
    # Split the 8-dim space into 3^4 = 81 sub-cubes by fixing the first 4 coords.
    # Within each sub-cube, solve a 4-dim covering problem using the 9-word tetracode,
    # then apply cross-subcube pruning.

    # The 4-dim ternary tetracode: 9 codewords covering all 3^4=81 words within radius 1.
    # Construct via a (4,9) ternary code: rows of a 4x9 matrix where each column is
    # a distinct vector and the code is a covering code of radius 1.
    # A simple construction: use the 9 codewords as the 9 points of an affine plane AG(2,3)
    # mapped to 4 coords: (x, y, x+y, x+2y) for x,y in {0,1,2}.
    tetracode_4d = []
    for x in range(3):
        for y in range(3):
            tetracode_4d.append([x, y, (x + y) % 3, (x + 2 * y) % 3])
    # Verify: 9 words, each covers 1 + 2*4 = 9 words; 9*9=81 = 3^4. Good.

    def partition_construct(deadline):
        """Build a full 8-dim cover via sub-cube tetracode + cross pruning."""
        # For each of the 81 sub-cubes (first 4 coords fixed), place a shifted tetracode
        # in the last 4 coords. To get diversity, we permute/shift the tetracode per sub-cube.
        selected = []
        used = set()
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    for d in range(3):
                        # Shift the tetracode by (a,b,c,d) in the last 4 coords
                        shift = (a + b + c + d) % 3
                        for tc in tetracode_4d:
                            word = [a, b, c, d,
                                    (tc[0] + shift) % 3,
                                    (tc[1] + (shift + 1) % 3),
                                    (tc[2] + (shift + 2) % 3),
                                    (tc[3] + shift) % 3]
                            idx = digits_to_idx(word)
                            if idx not in used:
                                selected.append(idx)
                                used.add(idx)
        return selected

    # ---------------- Dual-guided greedy set cover ----------------
    def dual_guided_greedy(selected, deadline):
        cover_count = np.zeros(total, dtype=np.int32)
        for c in selected:
            cover_count[balls[c]] += 1

        uncovered_mask = (cover_count == 0)

        while np.any(uncovered_mask):
            if time.time() > deadline - 10.0:
                break

            # Hardness: 1/cover_count for covered words, high for uncovered
            hardness = np.zeros(total, dtype=np.float64)
            nonzero = cover_count > 0
            hardness[nonzero] = 1.0 / cover_count[nonzero].astype(np.float64)
            hardness[~nonzero] = 2.0  # Uncovered get priority

            scores = hardness[balls].sum(axis=1)

            # Only consider candidates that cover at least one uncovered word
            covers_uncovered = uncovered_mask[balls].any(axis=1)
            if not np.any(covers_uncovered):
                break
            scores[~covers_uncovered] = -1

            max_score = np.max(scores)
            if max_score <= 0:
                break
            candidates = np.where(scores == max_score)[0]
            best_idx = candidates[np.random.randint(len(candidates))]

            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered_mask = (cover_count == 0)

        return selected, cover_count

    # ---------------- Local search ----------------
    def local_search(selected, cover_count, deadline):
        # Removal pass
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected

        # Swap optimization
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    improved = True
                    continue

                critical_mask = (cover_count[balls[c]] == 1)
                critical_words = balls[c][critical_mask]

                if len(critical_words) == 0:
                    cover_count[balls[c]] -= 1
                    improved = True
                    continue

                candidate_mask = np.ones(total, dtype=bool)
                for cw in critical_words:
                    candidate_mask &= ball_membership[:, cw]

                valid_candidates = np.where(candidate_mask)[0]
                valid_candidates = valid_candidates[valid_candidates != c]

                if len(valid_candidates) > 0:
                    c_prime = valid_candidates[np.random.randint(len(valid_candidates))]
                    cover_count[balls[c]] -= 1
                    cover_count[balls[c_prime]] += 1
                    new_selected.append(c_prime)
                    improved = True
                else:
                    new_selected.append(c)
            selected = new_selected

        # 2-Exchange: Remove 2, Add 1
        if len(selected) > 450 and time.time() < deadline - 8.0:
            improved_2ex = True
            while improved_2ex and time.time() < deadline - 5.0:
                improved_2ex = False
                new_selected = list(selected)
                indices = list(range(len(new_selected)))
                np.random.shuffle(indices)

                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        if time.time() > deadline - 5.0:
                            break
                        c1 = new_selected[indices[i]]
                        c2 = new_selected[indices[j]]

                        words_c1 = balls[c1]
                        words_c2 = balls[c2]
                        set_c2 = set(words_c2.tolist())

                        to_cover = set()
                        for w in words_c1:
                            if cover_count[w] == 1:
                                to_cover.add(w)
                        for w in words_c2:
                            if cover_count[w] == 1:
                                to_cover.add(w)
                        for w in words_c1:
                            if w in set_c2 and cover_count[w] == 2:
                                to_cover.add(w)

                        if not to_cover:
                            cover_count[balls[c1]] -= 1
                            cover_count[balls[c2]] -= 1
                            new_selected = [c for c in new_selected if c != c1 and c != c2]
                            improved_2ex = True
                            break

                        candidate_mask = np.ones(total, dtype=bool)
                        for w in to_cover:
                            candidate_mask &= ball_membership[:, w]
                        valid_candidates = np.where(candidate_mask)[0]
                        valid_candidates = valid_candidates[valid_candidates != c1]
                        valid_candidates = valid_candidates[valid_candidates != c2]

                        if len(valid_candidates) > 0:
                            c3 = valid_candidates[np.random.randint(len(valid_candidates))]
                            cover_count[balls[c1]] -= 1
                            cover_count[balls[c2]] -= 1
                            cover_count[balls[c3]] += 1
                            new_selected = [c for c in new_selected if c != c1 and c != c2]
                            new_selected.append(c3)
                            improved_2ex = True
                            break
                    if improved_2ex:
                        break
                if improved_2ex:
                    selected = new_selected

        return selected, cover_count

    best_size = total
    best_set = None

    for iteration in range(50):
        if time.time() > deadline - 5.0:
            break

        # Mixed initialization: partition construction, random, structured
        if iteration % 3 == 0:
            # Partition-based recursive construction
            selected = partition_construct(deadline)
        elif iteration % 3 == 1:
            # Pure random init
            selected = list(np.random.choice(total, size=500, replace=False))
        else:
            # Structured init
            selected = []
            for a in range(3):
                for b in range(3):
                    for c in range(3):
                        word = [a, b, c, a, b, c, a, b]
                        selected.append(digits_to_idx(word))
            for _ in range(200):
                idx = np.random.randint(0, total)
                if idx not in selected:
                    selected.append(idx)

        # Complete the cover using dual-guided greedy
        selected, cover_count = dual_guided_greedy(selected, deadline)

        if np.any(cover_count == 0):
            continue  # Invalid due to timeout

        # Local search
        selected, cover_count = local_search(selected, cover_count, deadline)

        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected[:]

        if best_size <= 486:
            break

    if best_set is None:
        selected, cover_count = dual_guided_greedy([], deadline)
        best_set = selected

    result = [idx_to_digits(cw) for cw in best_set]

    seen = set()
    unique_result = []
    for cw in result:
        t = tuple(cw)
        if t not in seen:
            seen.add(t)
            unique_result.append(cw)

    return unique_result
