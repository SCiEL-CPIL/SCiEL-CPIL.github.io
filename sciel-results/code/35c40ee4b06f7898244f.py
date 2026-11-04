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

    # Precompute balls (words within Hamming distance <= radius)
    # For radius=1, each ball has 1 + n*(q-1) words
    balls = []
    for w in all_words:
        w_idx = word_to_idx[w]
        neighbors = [w_idx]
        for i in range(n):
            for d in range(q):
                if d != w[i]:
                    new_w = list(w)
                    new_w[i] = d
                    neighbors.append(word_to_idx[tuple(new_w)])
        balls.append(neighbors)

    # Greedy set cover
    uncovered = np.ones(total, dtype=bool)
    covered_count = total
    chosen = []

    while covered_count > 0 and time.time() < deadline:
        best_idx = -1
        best_gain = 0
        for i in range(total):
            gain = 0
            for idx in balls[i]:
                if uncovered[idx]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_gain == 0:
            break
        chosen.append(best_idx)
        for idx in balls[best_idx]:
            if uncovered[idx]:
                uncovered[idx] = False
                covered_count -= 1

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

    chosen = _prune(chosen, balls)

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

    chosen = _local_search(chosen, balls, deadline)
    chosen = _prune(chosen, balls)

    # Final validity check
    if len(chosen) > 0:
        covered = np.zeros(total, dtype=bool)
        for w_idx in chosen:
            for j in balls[w_idx]:
                covered[j] = True
        if not np.all(covered):
            # Fallback
            if q == 3 and n >= 4:
                _TETRA = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
                return [h + list(tail) for h in _TETRA
                        for tail in itertools.product(range(q), repeat=n - 4)]
            return [list(w) for w in all_words]

    return [list(all_words[i]) for i in chosen]
