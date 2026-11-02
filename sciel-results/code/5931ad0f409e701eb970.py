import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)

    # Fallbacks for other parameters
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return [list(c) for c in itertools.product(range(q), repeat=n)]


def _luby_sequence():
    """Infinite Luby sequence for set-cover greedy."""
    m = 0
    while True:
        while m & 1 == 0:
            yield m
            m += 1
        yield m
        m += 1


def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561

    # Precompute balls (neighborhoods of radius 1)
    weights = [3 ** (n - 1 - i) for i in range(n)]
    balls = np.zeros((total, 1 + 2 * n), dtype=np.int32)
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

    # Precompute ball membership: ball_membership[v, w] = True if w in ball(v)
    ball_membership = np.zeros((total, total), dtype=bool)
    for w in range(total):
        ball_membership[w, balls[w]] = True

    # Precompute digit representations for Hamming distance
    digit_rep = np.zeros((total, n), dtype=np.int8)
    for w in range(total):
        temp = w
        for i in range(n - 1, -1, -1):
            digit_rep[w, i] = temp % 3
            temp //= 3

    def hamming_dist(a, b):
        return int(np.sum(digit_rep[a] != digit_rep[b]))

    def greedy_construct(seed):
        """Randomized greedy with Luby-sequence diversity bias."""
        rng = np.random.RandomState(seed)
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        luby_iter = _luby_sequence()
        luby_val = 0

        while np.any(uncovered_mask):
            if time.time() > deadline - 12.0:
                return None, None

            # Count coverage for each candidate among uncovered words
            counts = uncovered_mask[balls].sum(axis=1)
            max_count = np.max(counts)
            if max_count == 0:
                break
            candidates = np.where(counts == max_count)[0]

            if len(candidates) > 1:
                # Luby sequence with diversity bias
                if luby_val < 2:
                    # Pick randomly
                    best_idx = candidates[rng.randint(len(candidates))]
                else:
                    # Pick the candidate farthest from recently selected
                    recent = selected[-min(10, len(selected)):]
                    if recent:
                        min_dists = np.array([min(hamming_dist(cand, s) for s in recent)
                                              for cand in candidates])
                        # Soft-max weighting
                        max_d = np.max(min_dists)
                        weights_arr = np.exp((min_dists - max_d) / 1.5)
                        weights_arr /= weights_arr.sum()
                        best_idx = candidates[rng.choice(len(candidates), p=weights_arr)]
                    else:
                        best_idx = candidates[rng.randint(len(candidates))]
                luby_val += 1
            else:
                best_idx = candidates[0]

            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered_mask = (cover_count == 0)

        if np.any(uncovered_mask):
            return None, None
        return selected, cover_count

    # --- Phase 1: Multiple greedy restarts ---
    best_size = total
    best_set = None
    best_cc = None

    for restart in range(5):
        if time.time() > deadline - 20.0:
            break
        sel, cc = greedy_construct(restart * 7919 + 13)
        if sel is None:
            continue
        if len(sel) < best_size:
            best_size = len(sel)
            best_set = sel
            best_cc = cc.copy()

    if best_set is None:
        # Deterministic fallback
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        while np.any(uncovered_mask):
            counts = uncovered_mask[balls].sum(axis=1)
            max_count = np.max(counts)
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[0]
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered_mask = (cover_count == 0)
        best_set = selected
        best_cc = cover_count.copy()
        best_size = len(selected)

    # --- Phase 2: Local search with simulated annealing ---
    selected = best_set.copy()
    cover_count = best_cc.copy()
    current_size = len(selected)

    # Temperature schedule
    T_start = 5.0
    T_end = 0.1
    cool_rate = 0.97

    def try_remove_redundant(sel, cc):
        """Remove all redundant codewords."""
        new_sel = []
        changed = False
        for c in sel:
            if np.min(cc[balls[c]]) > 1:
                cc[balls[c]] -= 1
                changed = True
            else:
                new_sel.append(c)
        return new_sel, cc, changed

    def try_swap_improve(sel, cc):
        """Try 1-swap improvements with scoring."""
        improved = False
        new_sel = []
        for c in sel:
            if np.min(cc[balls[c]]) > 1:
                cc[balls[c]] -= 1
                improved = True
                continue

            critical_mask = (cc[balls[c]] == 1)
            critical_words = balls[c][critical_mask]

            if len(critical_words) == 0:
                cc[balls[c]] -= 1
                improved = True
                continue

            # Find candidates that cover all critical words
            candidate_mask = np.ones(total, dtype=bool)
            for cw in critical_words:
                candidate_mask &= ball_membership[:, cw]

            valid_candidates = np.where(candidate_mask)[0]
            valid_candidates = valid_candidates[valid_candidates != c]

            if len(valid_candidates) > 0:
                # Score by how many additional words they cover
                scores = np.zeros(len(valid_candidates), dtype=np.int32)
                for i, cp in enumerate(valid_candidates):
                    extra_cover = np.sum(cc[balls[cp]] == 1) - len(critical_words)
                    scores[i] = extra_cover

                best_i = np.argmax(scores)
                c_prime = valid_candidates[best_i]

                cc[balls[c]] -= 1
                cc[balls[c_prime]] += 1
                new_sel.append(c_prime)
                improved = True
            else:
                new_sel.append(c)

        return new_sel, cc, improved

    def try_2_exchange(sel, cc, rng, max_trials=500):
        """Try 2-exchange: remove 2 codewords, add 1. Returns (new_sel, new_cc, size_delta)."""
        if len(sel) < 3:
            return None, None, 0

        for _ in range(max_trials):
            if time.time() > deadline - 5.0:
                return None, None, 0

            idx1 = rng.randint(len(sel))
            idx2 = rng.randint(len(sel))
            if idx1 == idx2:
                continue

            c1 = sel[idx1]
            c2 = sel[idx2]

            # Temporarily remove c1 and c2
            cc[balls[c1]] -= 1
            cc[balls[c2]] -= 1

            uncovered = np.where(cc == 0)[0]

            if len(uncovered) == 0:
                # Both can be removed!
                new_sel = [c for i, c in enumerate(sel) if i != idx1 and i != idx2]
                return new_sel, cc, -2

            if len(uncovered) <= 8:
                # Find a single codeword that covers all uncovered words
                candidate_mask = np.ones(total, dtype=bool)
                for uw in uncovered:
                    candidate_mask &= ball_membership[:, uw]

                valid_candidates = np.where(candidate_mask)[0]
                valid_candidates = valid_candidates[
                    (valid_candidates != c1) & (valid_candidates != c2)]

                if len(valid_candidates) > 0:
                    # Score by coverage
                    scores = np.zeros(len(valid_candidates), dtype=np.int32)
                    for i, cp in enumerate(valid_candidates):
                        scores[i] = np.sum(cc[balls[cp]] == 0)  # words it uniquely covers
                    best_i = np.argmax(scores)
                    c3 = valid_candidates[best_i]

                    cc[balls[c3]] += 1
                    new_sel = [c for i, c in enumerate(sel) if i != idx1 and i != idx2]
                    new_sel.append(c3)
                    return new_sel, cc, -1

            # Revert
            cc[balls[c1]] += 1
            cc[balls[c2]] += 1

        return None, None, 0

    # Main local search loop
    T = T_start
    rng = np.random.RandomState(42)

    while time.time() < deadline - 5.0:
        # Phase A: Remove redundant
        selected, cover_count, changed = try_remove_redundant(selected, cover_count)
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()

        # Phase B: Try 1-swap
        if time.time() < deadline - 5.0:
            new_sel, new_cc, improved = try_swap_improve(selected, cover_count)
            if improved:
                selected = new_sel
                cover_count = new_cc
                selected, cover_count, _ = try_remove_redundant(selected, cover_count)
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_set = selected.copy()

        # Phase C: Try 2-exchange (structural move)
        if len(selected) > 420 and time.time() < deadline - 5.0:
            new_sel, new_cc, size_delta = try_2_exchange(selected, cover_count, rng,
                                                         max_trials=300)
            if new_sel is not None and size_delta < 0:
                selected = new_sel
                cover_count = new_cc
                selected, cover_count, _ = try_remove_redundant(selected, cover_count)
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_set = selected.copy()

        # Phase D: Simulated annealing perturbation
        if len(selected) > 420 and time.time() < deadline - 5.0:
            # Randomly swap a codeword with a random candidate
            idx = rng.randint(len(selected))
            c_old = selected[idx]
            c_new = rng.randint(total)

            if c_new != c_old:
                # Check if swap maintains coverage
                cover_count[balls[c_old]] -= 1
                cover_count[balls[c_new]] += 1

                uncovered = np.sum(cover_count == 0)

                if uncovered == 0:
                    # Accept (coverage maintained)
                    selected[idx] = c_new
                else:
                    # Simulated annealing: accept with probability exp(-delta/T)
                    delta = float(uncovered)
                    if rng.random() < np.exp(-delta / T):
                        selected[idx] = c_new
                    else:
                        # Revert
                        cover_count[balls[c_new]] -= 1
                        cover_count[balls[c_old]] += 1

        # Cool down
        T *= cool_rate
        if T < T_end:
            T = T_end

        if best_size <= 403:
            break

    # --- Phase 3: Final polish ---
    if best_set is not None:
        selected = best_set.copy()
        cover_count = np.zeros(total, dtype=np.int32)
        for c in selected:
            cover_count[balls[c]] += 1

        for _ in range(10):
            if time.time() > deadline - 1.0:
                break

            # Remove redundant
            changed = True
            while changed and time.time() < deadline - 1.0:
                changed = False
                new_selected = []
                for c in selected:
                    if np.min(cover_count[balls[c]]) > 1:
                        cover_count[balls[c]] -= 1
                        changed = True
                    else:
                        new_selected.append(c)
                selected = new_selected

            # Try swaps
            if time.time() < deadline - 1.0:
                new_selected, cover_count, improved = try_swap_improve(selected, cover_count)
                if improved:
                    selected = new_selected

            # Try 2-exchange
            if len(selected) > 420 and time.time() < deadline - 1.0:
                new_sel, new_cc, size_delta = try_2_exchange(selected, cover_count, rng,
                                                             max_trials=200)
                if new_sel is not None and size_delta < 0:
                    selected = new_sel
                    cover_count = new_cc

        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()

    # Convert to output format
    result = []
    for w in best_set:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)

    return result
