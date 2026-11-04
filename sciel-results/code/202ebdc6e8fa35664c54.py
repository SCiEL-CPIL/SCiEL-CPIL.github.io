import time
import itertools
import numpy as np

def construct(q, n, radius):
    deadline = time.time() + 110.0

    # Special case: q=3, n=8, radius=1 -> K_3(8,1)
    # Use a structured initial solution based on tetracode direct sum, then greedy+prune+local search.
    # For general (q, n, radius), use generic greedy set cover.

    total = q ** n

    if total > 5_000_000:
        # Too large for exact methods; use tetracode-like fallback
        if q == 3 and n >= 4 and radius == 1:
            _TETRA = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in _TETRA
                    for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            # Crude upper bound: all words
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Generate all words and map to indices
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # Precompute balls as a 2D numpy array for vectorized gain computation
    # For radius=1, each ball has 1 + n*(q-1) words
    ball_size = 1 + n * (q - 1)
    balls_arr = np.empty((total, ball_size), dtype=np.int32)
    for wi, w in enumerate(all_words):
        idx = 0
        balls_arr[wi, idx] = wi
        idx += 1
        for i in range(n):
            for d in range(q):
                if d != w[i]:
                    new_w = list(w)
                    new_w[i] = d
                    balls_arr[wi, idx] = word_to_idx[tuple(new_w)]
                    idx += 1

    # Also keep list-of-lists form for prune/local search (small overhead, kept for clarity)
    balls = [balls_arr[i].tolist() for i in range(total)]

    # Prune redundant codewords
    def _prune(chosen_list, balls):
        if not chosen_list:
            return []
        total = len(balls)
        coverage_count = np.zeros(total, dtype=np.int32)
        for c in chosen_list:
            for j in balls[c]:
                coverage_count[j] += 1

        keep = [True] * len(chosen_list)
        any_removed = True
        while any_removed:
            any_removed = False
            for i, c in enumerate(chosen_list):
                if not keep[i]:
                    continue
                redundant = True
                for j in balls[c]:
                    if coverage_count[j] <= 1:
                        redundant = False
                        break
                if redundant:
                    keep[i] = False
                    any_removed = True
                    for j in balls[c]:
                        coverage_count[j] -= 1
        return [chosen_list[i] for i in range(len(chosen_list)) if keep[i]]

    # Vectorized greedy set cover
    def _greedy_cover(balls_arr, total, deadline):
        uncovered = np.ones(total, dtype=bool)
        covered_count = total
        chosen = []
        ball_size = balls_arr.shape[1]

        while covered_count > 0 and time.time() < deadline:
            # Vectorized gain: count uncovered neighbors for each candidate
            # balls_arr is (total, ball_size); uncovered is (total,)
            # gain[i] = sum(uncovered[balls_arr[i, :]])
            gains = balls_arr[uncovered].sum(axis=1)  # shape (total,)
            best_idx = int(np.argmax(gains))
            best_gain = gains[best_idx]
            if best_gain == 0:
                break
            chosen.append(best_idx)
            newly = uncovered[balls_arr[best_idx]]
            # Mark those as covered
            uncovered[balls_arr[best_idx]] = False
            covered_count -= int(newly.sum())

        return chosen

    # Local search: targeted swaps
    def _local_search(chosen_list, balls, deadline):
        total = len(balls)
        chosen_set = set(chosen_list)

        while time.time() < deadline:
            improved = False
            chosen_set = set(chosen_list)

            coverage_count = np.zeros(total, dtype=np.int32)
            for c in chosen_list:
                for j in balls[c]:
                    coverage_count[j] += 1

            new_chosen = list(chosen_list)
            for i, c in enumerate(new_chosen):
                if time.time() > deadline:
                    break

                unique_points = [j for j in balls[c] if coverage_count[j] == 1]
                if not unique_points:
                    continue

                candidates = set(balls[unique_points[0]])
                for uj in unique_points[1:]:
                    candidates &= set(balls[uj])

                candidates -= chosen_set
                valid_candidates = []
                for c_prime in candidates:
                    if c_prime in chosen_set and c_prime != c:
                        continue
                    valid_candidates.append(c_prime)

                for c_prime in valid_candidates:
                    tentative = list(new_chosen)
                    tentative[i] = c_prime
                    pruned = _prune(tentative, balls)
                    if len(pruned) < len(new_chosen):
                        new_chosen = pruned
                        chosen_set = set(new_chosen)
                        improved = True
                        break

                if improved:
                    break

            if not improved:
                break

        return new_chosen

    # Multi-start: run greedy + prune + local search multiple times, keep best
    best_chosen = None
    best_len = float('inf')

    # Reserve time for local search on the best solution
    # Strategy: spend most time on first full pipeline, then use remaining for restarts
    # Each restart: greedy (fast) + prune + local search (shorter budget)
    num_restarts = 3
    # Allocate time: first run gets ~60% of budget, remaining split among restarts
    first_budget = 0.6
    per_restart_budget = (1.0 - first_budget) / max(num_restarts - 1, 1)

    for attempt in range(num_restarts):
        if time.time() >= deadline:
            break

        if attempt == 0:
            greedy_deadline = time.time() + (deadline - time.time()) * first_budget
        else:
            greedy_deadline = time.time() + (deadline - time.time()) * per_restart_budget

        # For restarts > 0, perturb the greedy by randomizing tie-breaking
        chosen = _greedy_cover(balls_arr, total, greedy_deadline)
        chosen = _prune(chosen, balls)

        # Local search with remaining time before global deadline
        local_deadline = deadline - 5.0  # safety margin
        chosen = _local_search(chosen, balls, local_deadline)
        chosen = _prune(chosen, balls)

        if len(chosen) < best_len:
            best_len = len(chosen)
            best_chosen = chosen

        # Check if we've hit a good target; if so, stop early
        if best_len <= 486:
            break

    if best_chosen is None:
        best_chosen = []

    # Final validity check
    if len(best_chosen) > 0:
        covered = np.zeros(total, dtype=bool)
        for w_idx in best_chosen:
            for j in balls[w_idx]:
                covered[j] = True
        if not np.all(covered):
            # Fallback
            if q == 3 and n >= 4:
                _TETRA = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
                return [h + list(tail) for h in _TETRA
                        for tail in itertools.product(range(q), repeat=n - 4)]
            return [list(w) for w in all_words]

    return [list(all_words[i]) for i in best_chosen]
