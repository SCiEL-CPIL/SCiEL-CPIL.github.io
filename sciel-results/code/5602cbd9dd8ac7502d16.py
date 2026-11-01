import time
import numpy as np
import random
from collections import defaultdict
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)

    # Fallback for other cases
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17

    # Precompute balls efficiently
    weights = [3 ** (n - 1 - i) for i in range(n)]

    # Build balls array: balls[w] = [w, neighbors...]
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
                balls[w, idx] = w + (new_digit - digit) * weight
                idx += 1

    # Build inverse mapping: for each word w, which ball centers cover it
    inv = [defaultdict(int) for _ in range(total)]
    for c in range(total):
        for w in balls[c]:
            inv[w][c] += 1
    inv_lists = [list(inv[w].keys()) for w in range(total)]

    # Precompute ball membership as a set for fast swap safety checks
    ball_sets = [set(balls[c].tolist()) for c in range(total)]

    best_size = total
    best_set = None

    # Use multiple random restarts with greedy + pruning + local search
    np.random.seed(42)
    random.seed(42)

    while time.time() < deadline - 5.0:
        # Randomized greedy construction with incremental updates
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_count = total

        # For each candidate center, track how many uncovered words it covers
        cand_count = np.zeros(total, dtype=np.int32)
        for c in range(total):
            cand_count[c] = ball_size

        # Track which centers are still "active" (not yet selected)
        active = np.ones(total, dtype=bool)

        while uncovered_count > 0:
            # Find the best candidate among active centers
            active_counts = np.where(active, cand_count, -1)
            max_count = np.max(active_counts)

            if max_count <= 0:
                break

            # Pick randomly among max-count active candidates for diversity
            candidates = np.where((active_counts == max_count) & active)[0]
            if len(candidates) == 0:
                break
            best_idx = candidates[np.random.randint(len(candidates))]

            selected.append(best_idx)
            active[best_idx] = False

            # Update cover counts and candidate counts
            for w in balls[best_idx]:
                if cover_count[w] == 0:
                    uncovered_count -= 1
                    for c in inv_lists[w]:
                        if active[c]:
                            cand_count[c] -= 1
                cover_count[w] += 1

            if time.time() > deadline - 10.0:
                break

        # Check if valid
        if uncovered_count > 0:
            continue

        # Pruning: iteratively remove codewords that are not needed (forward pass)
        changed = True
        while changed and time.time() < deadline - 4.0:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected

        # Pruning: reverse-order pass to escape removal dead-ends
        if time.time() < deadline - 4.0:
            changed = True
            while changed and time.time() < deadline - 4.0:
                changed = False
                new_selected = []
                for c in reversed(selected):
                    if np.min(cover_count[balls[c]]) > 1:
                        cover_count[balls[c]] -= 1
                        changed = True
                    else:
                        new_selected.append(c)
                selected = list(reversed(new_selected))

        # Local search: try to replace pairs of codewords with a single one
        # or replace a codeword with a better one
        if time.time() < deadline - 4.0:
            selected_set = set(selected)
            improved = True
            while improved and time.time() < deadline - 4.0:
                improved = False
                # Try removing each codeword if possible
                for i, c in enumerate(list(selected)):
                    if np.min(cover_count[balls[c]]) > 1:
                        cover_count[balls[c]] -= 1
                        selected.remove(c)
                        selected_set.discard(c)
                        improved = True

                # Try swapping: remove c1, add c2 if it improves coverage
                if time.time() > deadline - 8.0:
                    break

                for i in range(len(selected)):
                    if time.time() > deadline - 8.0:
                        break
                    c1 = selected[i]
                    # Words that would become uncovered if we remove c1
                    would_uncover = balls[c1][cover_count[balls[c1]] == 1]
                    if len(would_uncover) == 0:
                        cover_count[balls[c1]] -= 1
                        selected.remove(c1)
                        selected_set.discard(c1)
                        improved = True
                        break
                    else:
                        if len(would_uncover) <= 5:
                            for c2 in range(total):
                                if c2 in selected_set or c2 == c1:
                                    continue
                                if not all(w in ball_sets[c2] for w in would_uncover):
                                    continue
                                safe = True
                                for w in balls[c1]:
                                    if w not in would_uncover and cover_count[w] == 1:
                                        safe = False
                                        break
                                if safe:
                                    cover_count[balls[c1]] -= 1
                                    cover_count[balls[c2]] += 1
                                    selected.remove(c1)
                                    selected_set.discard(c1)
                                    selected.append(c2)
                                    selected_set.add(c2)
                                    improved = True
                                    break
                            if improved:
                                break

        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()

        # Early stop if we've found a very good solution and time is running low
        if time.time() > deadline - 8.0:
            break

    # Convert to list of digit lists
    if best_set is None:
        best_set = list(range(total))

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
