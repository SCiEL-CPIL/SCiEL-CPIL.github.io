"""Covering code K_3(7,1) via greedy set cover with bitset acceleration and local search."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

# Deadline: evaluator kills at 150s; we return well before that.
_DEADLINE = 140.0  # seconds


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + _DEADLINE

    # Total space
    total = q ** n

    # Precompute digit arrays for all words
    all_words = []
    for idx in range(total):
        digits = []
        x = idx
        for _ in range(n):
            digits.append(x % q)
            x //= q
        digits.reverse()
        all_words.append(tuple(digits))

    # Precompute powers for fast index computation
    powers = [q ** (n - 1 - i) for i in range(n)]

    # Precompute ball masks: for each center, a bitset of words within distance <= radius
    ball_masks = [0] * total
    for center_idx in range(total):
        center = all_words[center_idx]
        mask = 1 << center_idx
        if radius >= 1:
            for pos in range(n):
                for alt in range(q):
                    if alt == center[pos]:
                        continue
                    new_idx = center_idx + (alt - center[pos]) * powers[pos]
                    mask |= (1 << new_idx)
        ball_masks[center_idx] = mask

    # Greedy set cover
    all_mask = (1 << total) - 1
    uncovered = all_mask
    selected = []
    remaining = set(range(total))

    while uncovered != 0:
        if time.time() > deadline - 10:
            break

        best_count = -1
        best_idx = -1
        for c in remaining:
            covered = ball_masks[c] & uncovered
            cnt = covered.bit_count()
            if cnt > best_count:
                best_count = cnt
                best_idx = c

        if best_idx == -1 or best_count == 0:
            break

        selected.append(best_idx)
        uncovered &= ~ball_masks[best_idx]
        remaining.discard(best_idx)

    # If greedy didn't finish, fall back to direct sum construction
    if uncovered != 0:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Local search: try to remove redundant codewords
    # A codeword is redundant if its ball is fully covered by the union of other balls
    for _ in range(3):  # Multiple passes of removal
        if time.time() > deadline - 5:
            break
        removed_any = False
        for i in range(len(selected) - 1, -1, -1):
            # Check if selected[i] is redundant
            # Compute union of all other balls
            # This is expensive, so only do it if we have time
            if time.time() > deadline - 5:
                break
            # Quick check: is there any word in ball_masks[selected[i]] not covered by others?
            # We can approximate by checking a few random words
            # Actually, let's do it properly but efficiently
            # Build union of all other balls
            union_mask = 0
            for j in range(len(selected)):
                if j != i:
                    union_mask |= ball_masks[selected[j]]
            # If ball_masks[selected[i]] is subset of union_mask, it's redundant
            if ball_masks[selected[i]] & ~union_mask == 0:
                selected.pop(i)
                removed_any = True
                break  # Restart the loop since indices changed
        if not removed_any:
            break

    # Local search: try to swap codewords for better coverage
    # This is complex, so we'll just try random restarts of greedy if time permits
    best_solution = selected[:]
    best_size = len(selected)

    # Try a few random restarts with different orderings
    if time.time() < deadline - 20:
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            random.seed(42 + restart)
            # Shuffle the order of candidates
            shuffled = list(range(total))
            random.shuffle(shuffled)

            uncovered_r = all_mask
            selected_r = []
            remaining_r = set(shuffled)

            while uncovered_r != 0:
                if time.time() > deadline - 10:
                    break
                best_count = -1
                best_idx = -1
                for c in remaining_r:
                    covered = ball_masks[c] & uncovered_r
                    cnt = covered.bit_count()
                    if cnt > best_count:
                        best_count = cnt
                        best_idx = c
                if best_idx == -1 or best_count == 0:
                    break
                selected_r.append(best_idx)
                uncovered_r &= ~ball_masks[best_idx]
                remaining_r.discard(best_idx)

            if uncovered_r == 0 and len(selected_r) < best_size:
                best_solution = selected_r
                best_size = len(selected_r)

    # Convert best_solution indices back to lists of digits
    result = []
    for idx in best_solution:
        digits = []
        x = idx
        for _ in range(n):
            digits.append(x % q)
            x //= q
        digits.reverse()
        result.append(digits)

    return result
# EVOLVE-BLOCK-END
