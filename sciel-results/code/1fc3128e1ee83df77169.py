import time
import numpy as np
import random
from collections import defaultdict
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit

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

    # Precompute balls
    weights = [3 ** (n - 1 - i) for i in range(n)]
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

    # Build inverse mapping
    inv = [defaultdict(int) for _ in range(total)]
    for c in range(total):
        for w in balls[c]:
            inv[w][c] += 1
    inv_lists = [list(inv[w].keys()) for w in range(total)]

    # Precompute ball membership as sets for fast checks
    ball_sets = [set(balls[c].tolist()) for c in range(total)]

    # Precompute overlap counts: overlap[c1][c2] = number of words in both balls
    # This is expensive (6561^2 * 17) so we compute it smartly
    # Actually, let's compute on-demand with caching for the local search

    best_size = total
    best_set = None

    np.random.seed(42)
    random.seed(42)

    # Precompute for each word, the list of centers that cover it (for fast updates)
    # inv_lists[w] = list of centers c such that w is in ball[c]

    while time.time() < deadline - 5.0:
        # Randomized greedy construction
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_count = total

        # For each candidate center, track how many uncovered words it covers
        cand_count = np.zeros(total, dtype=np.int32)
        for c in range(total):
            cand_count[c] = ball_size

        active = np.ones(total, dtype=bool)

        while uncovered_count > 0 and time.time() < deadline - 10.0:
            active_counts = np.where(active, cand_count, -1)
            max_count = np.max(active_counts)

            if max_count <= 0:
                break

            candidates = np.where((active_counts == max_count) & active)[0]
            if len(candidates) == 0:
                break
            best_idx = candidates[np.random.randint(len(candidates))]

            selected.append(best_idx)
            active[best_idx] = False

            for w in balls[best_idx]:
                if cover_count[w] == 0:
                    uncovered_count -= 1
                    for c in inv_lists[w]:
                        if active[c]:
                            cand_count[c] -= 1
                cover_count[w] += 1

        if uncovered_count > 0:
            continue

        # Pruning: forward pass
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

        # Pruning: reverse pass
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

        # Local search with smart swap using precomputed overlap
        if time.time() < deadline - 4.0:
            selected_set = set(selected)
            improved = True
            while improved and time.time() < deadline - 4.0:
                improved = False

                # Try removing each codeword if possible
                for i in range(len(selected)):
                    c = selected[i]
                    if np.min(cover_count[balls[c]]) > 1:
                        cover_count[balls[c]] -= 1
                        selected.pop(i)
                        selected_set.discard(c)
                        improved = True
                        break

                # Try swapping: remove c1, add c2 if it improves
                if time.time() > deadline - 8.0:
                    break

                # For each codeword, find words that would become uncovered
                for i in range(len(selected)):
                    if time.time() > deadline - 8.0:
                        break
                    c1 = selected[i]
                    would_uncover = balls[c1][cover_count[balls[c1]] == 1]
                    if len(would_uncover) == 0:
                        cover_count[balls[c1]] -= 1
                        selected.pop(i)
                        selected_set.discard(c1)
                        improved = True
                        break
                    else:
                        # Find a replacement that covers all would_uncover words
                        # and doesn't create new uncovered words
                        if len(would_uncover) <= 8:
                            # Precompute which centers cover all words in would_uncover
                            # Use intersection of inv_lists for each word in would_uncover
                            candidate_centers = set(inv_lists[would_uncover[0]])
                            for w in would_uncover[1:]:
                                candidate_centers &= set(inv_lists[w])
                                if not candidate_centers:
                                    break

                            if candidate_centers:
                                # Remove c1 from candidates
                                candidate_centers.discard(c1)

                                # For each candidate, check if it's safe
                                # Safe means: for all words in ball[c1] not in would_uncover,
                                # cover_count[w] > 1 OR w is in ball[c2]
                                for c2 in candidate_centers:
                                    if c2 in selected_set:
                                        continue

                                    # Check safety
                                    safe = True
                                    for w in balls[c1]:
                                        if w not in would_uncover and cover_count[w] == 1:
                                            if w not in ball_sets[c2]:
                                                safe = False
                                                break
                                    if safe:
                                        cover_count[balls[c1]] -= 1
                                        cover_count[balls[c2]] += 1
                                        selected[i] = c2
                                        selected_set.discard(c1)
                                        selected_set.add(c2)
                                        improved = True
                                        break
                                if improved:
                                    break

        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()

        if time.time() > deadline - 8.0:
            break

    # Final high-precision polish pass on the best solution found
    if best_set is not None and time.time() < deadline - 2.0:
        selected = best_set.copy()
        selected_set = set(selected)
        cover_count = np.zeros(total, dtype=np.int32)
        for c in selected:
            cover_count[balls[c]] += 1

        # More aggressive local search with multiple passes
        for pass_num in range(3):
            if time.time() > deadline - 2.0:
                break
            improved = True
            while improved and time.time() < deadline - 2.0:
                improved = False

                # Try removing
                for i in range(len(selected)):
                    c = selected[i]
                    if np.min(cover_count[balls[c]]) > 1:
                        cover_count[balls[c]] -= 1
                        selected.pop(i)
                        selected_set.discard(c)
                        improved = True
                        break

                # Try swapping with broader search
                if time.time() > deadline - 3.0:
                    break

                for i in range(len(selected)):
                    if time.time() > deadline - 3.0:
                        break
                    c1 = selected[i]
                    would_uncover = balls[c1][cover_count[balls[c1]] == 1]
                    if len(would_uncover) == 0:
                        cover_count[balls[c1]] -= 1
                        selected.pop(i)
                        selected_set.discard(c1)
                        improved = True
                        break
                    else:
                        if len(would_uncover) <= 10:
                            candidate_centers = set(inv_lists[would_uncover[0]])
                            for w in would_uncover[1:]:
                                candidate_centers &= set(inv_lists[w])
                                if not candidate_centers:
                                    break

                            if candidate_centers:
                                candidate_centers.discard(c1)

                                # Sort candidates by how many additional words they cover
                                # (words in ball[c2] that are currently covered only once)
                                scored_candidates = []
                                for c2 in candidate_centers:
                                    if c2 in selected_set:
                                        continue
                                    # Count how many words in ball[c2] have cover_count == 1
                                    # and are not in would_uncover (those would be new uncovered)
                                    bonus = 0
                                    safe = True
                                    for w in balls[c2]:
                                        if w in would_uncover:
                                            continue
                                        if cover_count[w] == 1:
                                            safe = False
                                            break
                                        elif cover_count[w] > 1:
                                            bonus += 1
                                    if safe:
                                        scored_candidates.append((bonus, c2))

                                # Try the best candidates first
                                scored_candidates.sort(reverse=True)
                                for bonus, c2 in scored_candidates[:20]:  # Limit to top 20
                                    # Double-check safety
                                    safe = True
                                    for w in balls[c1]:
                                        if w not in would_uncover and cover_count[w] == 1:
                                            if w not in ball_sets[c2]:
                                                safe = False
                                                break
                                    if safe:
                                        cover_count[balls[c1]] -= 1
                                        cover_count[balls[c2]] += 1
                                        selected[i] = c2
                                        selected_set.discard(c1)
                                        selected_set.add(c2)
                                        improved = True
                                        break
                                if improved:
                                    break

        # Update best if improved
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected

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
