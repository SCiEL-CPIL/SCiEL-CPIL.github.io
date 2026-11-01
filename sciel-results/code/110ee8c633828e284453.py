import time
import numpy as np
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1_optimized(deadline, start_time)

    # Fallbacks for other cases
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n == 4 and radius == 1:
            return tetracode

    # Generic fallback: return all words (valid format, worst case)
    return [list(c) for c in itertools.product(range(q), repeat=n)]


def _construct_ternary_8_1_optimized(deadline, start_time):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17

    # Weights for positions (big-endian)
    weights = np.array([3 ** (n - 1 - i) for i in range(n)], dtype=np.int32)

    # Precompute balls for all words
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

    # Precompute: for each word, the set of codewords that cover it
    # We'll use a bitset approach for fast coverage checks
    # Actually, let's just use the balls array directly

    def greedy_construct(uncovered_mask, order_hint=None):
        """Greedy set cover with optional order hint for tie-breaking."""
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        current_uncovered = uncovered_mask.copy()

        while np.any(current_uncovered):
            # Count uncovered neighbors for each word
            counts = np.sum(current_uncovered[balls], axis=1)
            max_count = np.max(counts)

            if max_count == 0:
                break

            # Get candidates with max count
            candidates = np.where(counts == max_count)[0]

            if order_hint is not None and len(candidates) > 1:
                # Sort by order_hint position (lower index = earlier)
                order_arr = np.array(order_hint)
                candidate_positions = order_arr[candidates]
                best_pos = np.argmin(candidate_positions)
                best_idx = candidates[best_pos]
            else:
                best_idx = candidates[0]

            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            current_uncovered = (cover_count == 0)

        return selected, cover_count

    def prune(selected, cover_count):
        """Remove redundant codewords."""
        changed = True
        while changed:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
        return selected, cover_count

    def prune_reverse(selected, cover_count):
        """Remove redundant codewords in reverse order."""
        changed = True
        while changed:
            changed = False
            new_selected = []
            for c in reversed(selected):
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = list(reversed(new_selected))
        return selected, cover_count

    def local_search(selected, cover_count, max_iter=50):
        """Swap-based local search."""
        selected_set = set(selected)
        improved = True
        iter_count = 0

        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            new_selected = list(selected)
            new_selected_set = set(new_selected)

            for i in range(len(new_selected)):
                c = new_selected[i]
                for w in balls[c]:
                    if w in new_selected_set:
                        continue

                    # Try swap: remove c, add w
                    cover_count[balls[c]] -= 1
                    if np.any(cover_count == 0):
                        cover_count[balls[c]] += 1
                        continue

                    cover_count[balls[w]] += 1

                    # Check if we can remove another codeword
                    removable_d = None
                    for j in range(len(new_selected)):
                        if j == i:
                            continue
                        d = new_selected[j]
                        if np.min(cover_count[balls[d]]) > 1:
                            removable_d = d
                            break

                    if removable_d is not None:
                        cover_count[balls[removable_d]] -= 1
                        new_selected[i] = w
                        new_selected_set.remove(c)
                        new_selected_set.add(w)
                        new_selected.remove(removable_d)
                        new_selected_set.remove(removable_d)
                        improved = True
                        selected = new_selected
                        selected_set = new_selected_set
                        break
                    else:
                        cover_count[balls[w]] -= 1
                        cover_count[balls[c]] += 1

                    if improved:
                        break
                if improved:
                    break

        return selected, cover_count

    # Phase 1: Multiple greedy constructions with different tie-breaking
    best_selected = None
    best_cover = None
    best_size = float('inf')

    rng = random.Random(42)
    full_order = list(range(total))

    # Try several different orderings
    num_greedy = 8
    for g in range(num_greedy):
        if time.time() > deadline - 30.0:
            break

        if g == 0:
            order = None  # Default (smallest index)
        elif g == 1:
            order = list(reversed(full_order))  # Largest index first
        else:
            order = full_order[:]
            rng.shuffle(order)

        selected, cover_count = greedy_construct(np.ones(total, dtype=bool), order)
        selected, cover_count = prune(selected, cover_count)
        selected, cover_count = prune_reverse(selected, cover_count)

        if len(selected) < best_size:
            best_selected = list(selected)
            best_cover = cover_count.copy()
            best_size = len(selected)

    # Phase 2: Local search from best solution
    if best_selected is not None:
        selected, cover_count = local_search(best_selected, best_cover, max_iter=100)
        selected, cover_count = prune(selected, cover_count)
        selected, cover_count = prune_reverse(selected, cover_count)

        if len(selected) < best_size:
            best_selected = list(selected)
            best_cover = cover_count.copy()
            best_size = len(selected)

    # Phase 3: Perturbation and repair
    max_restarts = 15
    for r in range(max_restarts):
        if time.time() > deadline - 10.0:
            break

        selected = list(best_selected)
        cover_count = best_cover.copy()

        # Perturbation: remove k random codewords
        k = rng.randint(3, min(15, len(selected) // 5))
        if len(selected) > k:
            to_remove = rng.sample(selected, k)
            to_remove_set = set(to_remove)
            selected = [c for c in selected if c not in to_remove_set]

            # Recompute cover_count
            cover_count = np.zeros(total, dtype=np.int32)
            for c in selected:
                cover_count[balls[c]] += 1

            # Repair: add words to cover uncovered positions
            current_uncovered = (cover_count == 0)
            while np.any(current_uncovered):
                if time.time() > deadline - 15.0:
                    break
                counts = np.sum(current_uncovered[balls], axis=1)
                max_count = np.max(counts)
                if max_count == 0:
                    break
                candidates = np.where(counts == max_count)[0]
                best_idx = candidates[0]
                selected.append(best_idx)
                cover_count[balls[best_idx]] += 1
                current_uncovered = (cover_count == 0)

        # Prune
        selected, cover_count = prune(selected, cover_count)
        selected, cover_count = prune_reverse(selected, cover_count)

        # Local search
        selected, cover_count = local_search(selected, cover_count, max_iter=30)
        selected, cover_count = prune(selected, cover_count)

        # Accept if better
        if len(selected) < best_size:
            best_selected = list(selected)
            best_cover = cover_count.copy()
            best_size = len(selected)

    # Final pruning
    selected, cover_count = prune(best_selected, best_cover)
    selected, cover_count = prune_reverse(selected, cover_count)

    # Final verification
    final_cover = np.zeros(total, dtype=np.int32)
    for c in selected:
        final_cover[balls[c]] += 1

    if np.any(final_cover == 0):
        missing = np.where(final_cover == 0)[0]
        for m in missing:
            selected.append(m)
            final_cover[balls[m]] += 1

    # Convert to digit lists
    result = []
    for w in selected:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)

    return result
